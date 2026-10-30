import hashlib
import io
import gc
import logging
from pathlib import Path
from django.conf import settings
from PIL import Image

class SuryaExtractor:
    """Surya OCR/layout for page images; native parsers for Office and text.

    Preserve original pixels and source regions. Real vectors are generated
    separately by BGE and ColPali indexers; missing models are not simulated.
    """
    from threading import Lock
    _shared_ocr_predictor = None
    _shared_ocr_lock = Lock()

    def __init__(self, *, load_models=True):
        self.output_img_dir = Path(settings.MEDIA_ROOT) / 'extracted_images'
        self.output_img_dir.mkdir(parents=True, exist_ok=True)
        self.surya_layout_model = None
        if load_models:
            self._load_surya_models()

    def _load_surya_models(self):
        """
        Nạp toàn bộ bộ mô hình Surya Vision Suite (RecognitionPredictor) Singleton
        """
        if getattr(settings, 'RAG_INFERENCE_URL', ''):
            from ai_extract.services.contract_pipeline_service import RemoteLayoutPredictor
            self.surya_layout_model = RemoteLayoutPredictor(self)
            return
        if SuryaExtractor._shared_ocr_predictor is None:
            try:
                from .surya_inference_service import create_surya_predictor
                SuryaExtractor._shared_ocr_predictor = create_surya_predictor()
            except Exception as error:
                print(f"[Surya OCR] Model unavailable: {type(error).__name__}")
                return
        from ai_extract.services.contract_pipeline_service import LocalLayoutPredictor
        self._surya_ocr_predictor = SuryaExtractor._shared_ocr_predictor
        self._surya_ocr_lock = SuryaExtractor._shared_ocr_lock
        self.surya_layout_model = LocalLayoutPredictor(self)

    def extract_document(self, doc_file):
        """
        Main extraction router based on document category, tích hợp Hierarchical Semantic Chunker
        """
        category = doc_file.category
        file_path = doc_file.file.path
        doc_id = doc_file.id

        if category == 'pdf':
            raw_chunks = self._extract_pdf_pure_surya_vision(file_path, doc_id)
        elif category == 'image':
            raw_chunks = self._extract_image(file_path, doc_id, doc_file.file.url)
        elif category == 'docx':
            raw_chunks = self._extract_docx(file_path, doc_id)
        elif category == 'xlsx':
            from .office_parser import extract_spreadsheet
            raw_chunks = extract_spreadsheet(file_path, self.output_img_dir, doc_id)
        elif category == 'pptx':
            if Path(file_path).suffix.lower() != '.pptx':
                raise ValueError('Hãy chuyển slide sang PPTX trước khi trích xuất.')
            from .office_parser import extract_pptx
            raw_chunks = extract_pptx(file_path, self.output_img_dir, doc_id)
        else:
            raw_chunks = self._extract_text(file_path, doc_id)
            for chunk in raw_chunks:
                chunk.update(bbox=[], location_kind='line',
                             source_location=f"Dòng {chunk['chunk_id']}")

        # Streamed PDF pages have already been normalized by the A3 callback.
        # The caller returns those persisted page results, not a second copy.
        if category == 'pdf' and getattr(self, 'on_page_records', None):
            return raw_chunks

        # TÍCH HỢP BỘ PHÂN ĐOẠN PHÂN CẤP & LIÊN KẾT NGỮ NGHĨA (HIERARCHICAL CHUNKER)
        try:
            from ai_extract.services.document_structure_service import HierarchicalChunker
            doc_title = getattr(doc_file, 'original_name', None) or Path(file_path).name
            processed_chunks = HierarchicalChunker.process_hierarchical_chunks(raw_chunks, doc_title=doc_title)
            return processed_chunks
        except Exception as chunk_err:
            print(f"[HierarchicalChunker Notice] {chunk_err}")
            return raw_chunks


    def _draw_visual_bbox_crop(self, page_img_path: Path, bbox_norm: list, crop_filename: str) -> str:
        try:
            crop_save_path = self.output_img_dir / crop_filename

            with Image.open(page_img_path) as img:
                img = img.convert("RGB")
                w, h = img.size

                x_min = int((bbox_norm[0] / 1000.0) * w)
                y_min = int((bbox_norm[1] / 1000.0) * h)
                x_max = int((bbox_norm[2] / 1000.0) * w)
                y_max = int((bbox_norm[3] / 1000.0) * h)

                pad = 15
                crop_x0 = max(0, x_min - pad)
                crop_y0 = max(0, y_min - pad)
                crop_x1 = min(w, x_max + pad)
                crop_y1 = min(h, y_max + pad)

                if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                    return ""

                cropped = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))

                cropped.save(crop_save_path, "PNG")

                return f"{settings.MEDIA_URL}extracted_images/{crop_filename}"
        except Exception as e:
            print("[Draw BBox Error]", str(e))
            return ""

    def _ocr_png_image(self, pil_image: Image.Image) -> list:
        """
        Nhận diện ký tự trực tiếp từ ma trận điểm ảnh (Pixels) của ảnh PNG bằng OCR Engine.
        Tuyệt đối không phụ thuộc vào text layer của PDF.
        Trả về danh sách các dòng chữ kèm tọa độ bbox [x0, y0, x1, y1] trên ảnh PNG.
        """
        if getattr(settings, 'RAG_INFERENCE_URL', ''):
            from ai_extract.services.contract_pipeline_service import remote_page
            return remote_page(self,pil_image)['ocr_lines']
        try:
            from ai_extract.services.contract_pipeline_service import local_page
            if not getattr(self, '_surya_ocr_predictor', None):
                raise RuntimeError('Surya OCR is unavailable')
            return local_page(self, pil_image)['ocr_lines']
        except Exception as error:
            if getattr(self, 'strict', False):
                raise RuntimeError('A3 Surya OCR failed; no extracted text can be confirmed.') from error
            return []

    def _extract_chunks_from_page_image(
        self, page_img, page_num, doc_id, doc_filename, page_img_path,
        page_img_url, start_chunk_id=1,
    ):
        """Keep each Surya source region intact; never regroup by proximity."""
        ocr_lines = self._ocr_png_image(page_img)
        regions = []
        if self.surya_layout_model:
            try:
                layout = self.surya_layout_model([page_img])[0]
                regions = getattr(layout, 'bboxes', [])
            except Exception:
                if getattr(self, 'strict', False):
                    raise
        if any(not getattr(region, 'region_id', None) for region in regions):
            raise ValueError('Surya regions require source IDs; refresh the legacy OCR cache.')
        return self._chunks_from_surya_regions(
            regions, ocr_lines, page_img, page_num, doc_id,
            page_img_path, page_img_url, start_chunk_id,
        )

    def _chunks_from_surya_regions(self, regions, lines, image, page_num, doc_id,
                                  image_path, image_url, start_chunk_id):
        """A Surya layout region is already a chunk; never re-group its text."""
        from .document_structure_service import normalize_layout
        by_region = {}
        unbound = []
        region_ids = {region.region_id for region in regions}
        for line in lines:
            region_id = line.get('region_id')
            if not region_id:
                matches = [r for r in regions if list(r.bbox) == list(line['bbox'])]
                if len(matches) == 1: region_id = matches[0].region_id
            if region_id in region_ids:
                by_region.setdefault(region_id, []).append(line['text'])
            else: unbound.append(line)
        chunks = []
        def append(kind, text, bbox, confidence, metadata):
            metadata.setdefault('text_source','surya_ocr')
            normalized = [round(v/(image.width if i%2==0 else image.height)*1000,4) for i,v in enumerate(bbox)]
            number = start_chunk_id+len(chunks)
            crop_url = self._draw_visual_bbox_crop(image_path, normalized,
                f'crop_doc{doc_id}_p{page_num}_b{number}.png') if kind in {'figure','table'} else ''
            chunks.append(dict(chunk_id=number,layout_type=kind,text=text,bbox=normalized,page_number=page_num,
                has_image=bool(crop_url),image_url=crop_url or image_url,confidence=confidence,
                fallback_triggered=False,fallback_model=None,vector_dim=None,vector_sample=[],
                layout_source='surya_layout',chunk_policy='surya_layout_region',**metadata))
        caption_regions = [r for r in regions if normalize_layout(getattr(r, 'label', '')) in {'figure_caption', 'caption', 'table_caption'}]
        for region in regions:
            kind = normalize_layout(region.label)
            metadata = dict(surya_region_id=region.region_id,
                surya_parent_region_id=getattr(region,'parent_region_id',None),
                visual_depth=getattr(region,'visual_depth',0),
                ocr_status=getattr(region,'ocr_status','completed'),
                reading_order=getattr(region,'reading_order',None),
                reading_order_source=getattr(region,'reading_order_source','surya_legacy_order_unknown'),
                reading_order_scope=getattr(region,'reading_order_scope',getattr(region,'parent_region_id',None) or 'page'),
                reading_order_status=getattr(region,'reading_order_status','unresolved'),
                reading_order_unresolved=getattr(region,'reading_order_unresolved',True),
                source_region_order=getattr(region,'source_region_order',len(chunks)))
            for key in ('visual_ocr_status','visual_ocr_error','visual_ocr_candidate_failures','visual_ocr_candidate_rejections',
                        'heading_level','heading_source','hierarchy_level_status','table_rows','table_cells',
                        'table_header_origin','table_structure_status','ocr_validation'):
                if hasattr(region,key): metadata[key]=getattr(region,key)
            if kind == 'figure':
                in_image_lines = by_region.get(region.region_id, [])
                metadata['in_image_text'] = in_image_lines
                caption_text = ''
                linked=[cap for cap in caption_regions if getattr(cap,'parent_region_id',None)==region.region_id]
                if len(linked)==1:
                    caption_text=' '.join(by_region.get(linked[0].region_id,[])).strip()
                    if caption_text:
                        metadata['caption_text']=caption_text
                        metadata['linked_caption']=caption_text
                parts = []
                if caption_text:
                    parts.append(f"[{caption_text}]")
                else:
                    parts.append(f"[Hình ảnh kỹ thuật trang {page_num}]")
                if in_image_lines:
                    parts.append("Nhãn trong ảnh: " + ", ".join(in_image_lines))
                text = "\n".join(parts)
                # This is a UI description assembled by the application, not
                # an OCR quotation. Child OCR/caption regions remain separate.
                metadata.update(text_source='visual_display_label',display_only=True,display_text=text)
            else:
                text = '\n'.join(by_region.get(region.region_id,[]))
            append(kind,text,region.bbox,getattr(region,'confidence',0),metadata)
        # Legacy caches can have unbound OCR. Retain each source region alone,
        # with explicit uncertainty, rather than merging unrelated paragraphs.
        for line in unbound:
            append('paragraph',line['text'],line['bbox'],line.get('score',0),
                dict(ocr_status='completed',reading_order_unresolved=True))
        return chunks,start_chunk_id+len(chunks)

    def _extract_logged_page(self, *, total_pages, **kwargs):
        from ai_extract.services.inference_runtime_service import PipelineProgress
        with PipelineProgress(kwargs['doc_id']).stage('surya', page=kwargs['page_num'],
                                                      total_pages=total_pages) as details:
            records, counter = self._extract_chunks_from_page_image(**kwargs)
            details['blocks'] = len(records)
            cached = getattr(self, '_last_remote_page', getattr(self, '_last_local_page', None))
            details['cache_hit'] = bool(cached and cached[1].get('cache_hit'))
            metrics = cached[1].get('metrics', {}) if cached else {}
            if metrics:
                details.update(ocr_mode=metrics.get('ocr_mode'),
                               failed_regions=metrics.get('failed_regions',0),
                               model_elapsed_s=metrics.get('elapsed_s'),
                               generation_calls=len(metrics.get('generation_calls', [])),
                               crop_retries=sum(call.get('retries', 0) for call in metrics.get('generation_calls', [])))
                if metrics.get('visual_ocr'):
                    details['visual_ocr'] = {k:v for k,v in metrics['visual_ocr'].items() if k!='generation_calls'}
                details['body_cache_hit'] = bool(cached[1].get('body_cache_hit'))
                if metrics.get('reading_order'):
                    details['reading_order'] = metrics['reading_order']
            return records, counter

    def _extract_pdf_pure_surya_vision(self, file_path, doc_id):
        """Render every PDF page to pixels and extract only through Surya OCR/layout."""
        chunks = []
        chunk_counter = 1
        doc_filename = Path(file_path).name
        pdf_doc = None
        page_img = pix = sharp = None
        def release_page_buffers():
            nonlocal page_img,pix,sharp
            visual=getattr(self,'_visual_ocr_image',None)
            if visual is not None:visual.close()
            if page_img is not None:page_img.close()
            self._visual_ocr_image=None
            page_img=pix=sharp=None

        try:
            import fitz  # Sử dụng duy nhất để rasterize trang PDF thành PIL Image (PNG)
            pdf_doc = fitz.open(file_path)

            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]

                # 1. RASTERIZE TRANG PDF THÀNH ẢNH PNG (PIXELS THUẦN TÚY)
                pix = page.get_pixmap(dpi=150, colorspace=fitz.csRGB, alpha=False)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # Keep the existing body/layout frame; render sharper pixels
                # for text inside figures instead of enlarging tiny glyphs.
                sharp = page.get_pixmap(dpi=300, colorspace=fitz.csRGB, alpha=False)
                self._visual_ocr_image = Image.frombytes('RGB', [sharp.width, sharp.height], sharp.samples)

                # Lưu ảnh PNG chuẩn hóa vào thư mục media để phục vụ cắt BBox và hiển thị UI
                page_img_filename = f"pdf_page_{doc_id}_p{page_num+1}.png"
                page_img_path = self.output_img_dir / page_img_filename
                page_img.save(page_img_path, format="PNG")
                structure_path=self.output_img_dir/f'pdf_page_{doc_id}_p{page_num+1}_structure.png'
                self._visual_ocr_image.save(structure_path,format='PNG')
                page_img_url = f"{settings.MEDIA_URL}extracted_images/{page_img_filename}"
                if getattr(self, 'on_page_image', None):
                    self.on_page_image(page_num + 1, page_img_path, page_img.width, page_img.height,
                                       total_pages=len(pdf_doc))

                # 2. ĐỌC NỘI DUNG 100% TỪ PNG BẰNG OCR (NHẬN DIỆN TỪ ĐIỂM ẢNH)
                try:
                    page_chunks, chunk_counter = self._extract_logged_page(
                        total_pages=len(pdf_doc), page_img=page_img, page_num=page_num + 1,
                        doc_id=doc_id, doc_filename=doc_filename, page_img_path=page_img_path,
                        page_img_url=page_img_url, start_chunk_id=chunk_counter)
                except Exception as error:
                    if not getattr(self, 'on_page_error', None):
                        raise
                    # Keep successful pages. An unreadable page remains explicitly
                    # failed, never an empty successful OCR result.
                    self.on_page_error(page_num + 1, error)
                    release_page_buffers()
                    continue
                cached_response = getattr(self, '_last_remote_page', getattr(self, '_last_local_page', None))
                cache_hit = bool(cached_response and cached_response[1].get('cache_hit'))
                for record in page_chunks:
                    record.setdefault('text_source','surya_ocr')
                    record.update(layout_source='surya', ocr_cache_hit=cache_hit)
                    media_root=Path(settings.MEDIA_ROOT).resolve()
                    if structure_path.resolve().is_relative_to(media_root):
                        record['structure_image_path']=structure_path.resolve().relative_to(media_root).as_posix()
                if getattr(self,'retain_page_results',True):
                    chunks.extend(page_chunks)

                if getattr(self, 'on_page_records', None):
                    self.on_page_records(page_num + 1, page_chunks)
                release_page_buffers()

        except Exception as e:
            logging.getLogger(__name__).exception('Surya PDF extraction failed for %s',doc_id)
            if getattr(self, 'strict', False):
                raise
        finally:
            release_page_buffers()
            if pdf_doc is not None:
                pdf_doc.close()
            gc.collect()

        return chunks

    def _extract_image(self, file_path, doc_id, image_url=None):
        """
        Xử lý 100% file ảnh (PNG/JPG/TIFF...) theo luồng Pure Image Vision:
        1. Đọc trực tiếp ma trận điểm ảnh của ảnh tải lên.
        2. Nhận diện chữ trực tiếp từ ma trận điểm ảnh PNG bằng OCR Engine.
        3. Phân tích bố cục vùng và tạo chunks trực quan kèm BBox và vector.
        """
        chunks = []
        try:
            doc_filename = Path(file_path).name
            with Image.open(file_path) as img:
                page_img = img.convert("RGB")
                page_img_filename = f"img_doc_{doc_id}.png"
                page_img_path = self.output_img_dir / page_img_filename
                page_img.save(page_img_path, format="PNG")
                page_img_url = f"{settings.MEDIA_URL}extracted_images/{page_img_filename}"
                if getattr(self, 'on_page_image', None):
                    self.on_page_image(1, page_img_path, page_img.width, page_img.height, total_pages=1)

                page_chunks, _ = self._extract_logged_page(
                    total_pages=1,
                    page_img=page_img,
                    page_num=1,
                    doc_id=doc_id,
                    doc_filename=doc_filename,
                    page_img_path=page_img_path,
                    page_img_url=page_img_url,
                    start_chunk_id=1
                )
                chunks.extend(page_chunks)
        except Exception:
            raise

        return chunks

    def _extract_docx(self, file_path, doc_id):
        from .office_parser import extract_docx
        return extract_docx(file_path, self.output_img_dir, doc_id)

    def _extract_text(self, file_path, doc_id):
        """Text files have source lines, without fabricated page coordinates."""
        lines = Path(file_path).read_text(encoding='utf-8', errors='ignore').splitlines()
        return [dict(chunk_id=index, text=line.strip(), bbox=[], page_number=1,
                     layout_type='log_entry' if 'ERROR' in line or 'WARN' in line else 'paragraph',
                     has_image=False, image_url=None, vector_dim=None, vector_sample=[])
                for index, line in enumerate(lines, 1) if line.strip()]



def save_embedded_image(blob, directory, document_key):
    if len(blob) > 30 * 1024 * 1024:
        raise ValueError('Embedded image exceeds the 30 MB limit.')
    with Image.open(io.BytesIO(blob)) as image:
        image.verify()
        extension = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp', 'GIF': '.gif', 'TIFF': '.tiff', 'BMP': '.bmp'}.get(image.format)
        if not extension:
            raise ValueError('Embedded image format is not supported.')
    key = hashlib.sha256(blob).hexdigest()
    name = f'embedded_{hashlib.sha256(str(document_key).encode()).hexdigest()[:16]}_{key[:24]}{extension}'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/name
    path.write_bytes(blob)
    return path


def extraction_metadata(record, media_root):
    keys = ('heading_path', 'section_level', 'heading_level', 'heading_source', 'section_title',
            'structure_kind','hierarchy_level_status',
            'source_location', 'location_kind', 'cell_values', 'table_headers', 'table_row',
            'formulas', 'merged_ranges', 'linked_caption', 'visual_status', 'extractor_version', 'original_layout_type',
            'text_source', 'display_only', 'display_text', 'layout_source', 'ocr_cache_hit', 'table_values', 'table_rows', 'table_cells', 'table_structure_status', 'table_header_origin', 'table_id',
            'visual_ocr_status','visual_ocr_error','text_bbox_scope','visual_text_regions','visual_ocr_candidate_failures',
            'visual_ocr_candidate_rejections','visual_ocr_unreadable_lines',
            'surya_region_id','surya_parent_region_id','visual_depth','chunk_policy','ocr_status','ocr_validation',
            'reading_order','reading_order_source','reading_order_scope','reading_order_status',
            'reading_order_unresolved','source_region_order','structure_image_path',
            'caption_text','in_image_text','summary_description')
    metadata = {key: record[key] for key in keys if key in record}
    root = Path(media_root).resolve()
    relative = record.get('image_path')
    if not relative:
        from django.conf import settings
        from urllib.parse import unquote, urlparse
        url = urlparse(record.get('image_url') or '')
        if not url.scheme and url.path.startswith(settings.MEDIA_URL):
            relative = unquote(url.path[len(settings.MEDIA_URL):])
    if relative:
        path = (root/relative).resolve()
        if path.is_relative_to(root) and path.is_file():
            metadata['image_path'] = path.relative_to(root).as_posix()
    return metadata


def colpali_point_id(run_id, page_id, block_id):
    from uuid import uuid5, NAMESPACE_URL
    return str(uuid5(NAMESPACE_URL, f'contract-dl:colpali:{run_id}:{page_id}:block:{block_id}'))


def visual_sources(run, *, page_number=None):
    """Only confirmed figure blocks, cropped from rendered pixels; no page fallback."""
    import math
    from types import SimpleNamespace
    from .document_structure_service import normalize_layout
    root = Path(settings.MEDIA_ROOT).resolve()
    rows = run.pages
    if page_number is not None:
        rows = rows.filter(page_number=page_number)
    for page in rows.prefetch_related('blocks').order_by('page_number'):
        for block in page.blocks.all():
            if normalize_layout(block.layout_type) != 'figure':
                continue
            if page.image_path:
                source = (root/page.image_path).resolve()
                if not source.is_relative_to(root):
                    raise ValueError('Visual source outside media root')
                box = block.bbox
                if (len(box) != 4 or any(isinstance(v,bool) or not isinstance(v,(float,int))
                        or not math.isfinite(v) or not 0 <= v <= 1000 for v in box)
                        or box[0] >= box[2] or box[1] >= box[3]):
                    raise ValueError(f'Invalid figure bbox: block {block.pk}')
                relative = f'extracted_images/colpali_run_{run.id}_block_{block.pk}.png'
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    w,h = image.size
                    bounds = (math.floor(box[0]*w/1000), math.floor(box[1]*h/1000),
                              math.ceil(box[2]*w/1000), math.ceil(box[3]*h/1000))
                    image.convert('RGB').crop(bounds).save(target, format='PNG')
            else:
                relative = (block.structure_metadata or {}).get('image_path')
                if not relative:
                    raise ValueError(f'Missing figure image: block {block.pk}')
                source = (root/relative).resolve()
                if not source.is_relative_to(root):
                    raise ValueError('Visual source outside media root')
                # Office assets are already individual images, but always send PNG.
                relative = f'extracted_images/colpali_run_{run.id}_block_{block.pk}.png'
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    image.convert('RGB').save(target, format='PNG')
            yield SimpleNamespace(page=page, block=block, image_path=relative)
