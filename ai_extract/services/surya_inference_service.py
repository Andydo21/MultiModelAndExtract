"""Surya OCR inference and source block normalization without web ORM imports."""
import base64
import io
import json
import math
import os
import time
import subprocess
import atexit
from pathlib import Path
from threading import RLock
from urllib.request import urlopen
from urllib.parse import urlparse
from copy import copy
from concurrent.futures import ThreadPoolExecutor

_managed_backend = None
VISUAL_OCR_POLICY = 'surya-image-text-only-v22-single-pass'
BODY_OCR_POLICY = 'surya-high-resolution-v3-source-html'
TABLE_CELL_POLICY = 'physical-cell-surya-ocr-v18-model-spans'


def physical_table_grid(width, height, horizontal, vertical, column_boxes=None, spanning_boxes=(), *, atomic=False,row_segments=None,flexible_rows=False):
    """Decode observed separators, never text baselines, into physical cells.

    Production column counts come from the structure model; observed rules
    refine their boundaries. Missing rules imply spans on a visibly ruled axis.
    """
    from statistics import median
    tolerance=max(2,round(min(width,height)*.004))
    def cluster(values, distance=None):
        distance=tolerance if distance is None else distance
        groups=[]
        for value in sorted(values):
            if not groups or value-groups[-1][-1]>distance:groups.append([value])
            else:groups[-1].append(value)
        return [float(median(group)) for group in groups]
    def coverage(intervals,low,high):
        merged=[]
        for a,b in sorted((max(a,low),min(b,high)) for a,b in intervals if min(b,high)>max(a,low)):
            if merged and a<=merged[-1][1]+tolerance:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        return sum(b-a for a,b in merged)/max(1,high-low)
    if len(horizontal)<2:raise ValueError('No reliable physical row separators')
    left=min(line[0] for line in horizontal);right=max(line[2] for line in horizontal)
    x_tolerance=max(tolerance,round(width*.005))
    observed=cluster([line[0] for line in vertical if left+2*x_tolerance<line[0]<right-2*x_tolerance],x_tolerance)
    vlow=min((l[1] for l in vertical),default=0);vhigh=max((l[3] for l in vertical),default=height)
    def vertical_support(x):
        return coverage([(l[1],l[3]) for l in vertical if abs(l[0]-x)<=x_tolerance],vlow,vhigh)
    if column_boxes and 1<=len(column_boxes)<=64:
        boxes=sorted(column_boxes,key=lambda box:box[0])
        xs=[float(left),*[(a[2]+b[0])/2 for a,b in zip(boxes,boxes[1:])],float(right)]
        # A partial physical rule gives a sharper column boundary than the
        # approximate detector box. Only snap within its local column gap.
        ends=[v for line in horizontal for v in (line[0],line[2]) if left+tolerance<v<right-tolerance]
        for i in range(1,len(xs)-1):
            nearby=[x for x in observed if abs(x-xs[i])<min(xs[i]-xs[i-1],xs[i+1]-xs[i])*.2]
            if nearby:
                best=max(nearby,key=vertical_support)
                if vertical_support(best)>=.5:
                    xs[i]=best
                    continue
            candidates=[v for v in ends if abs(v-xs[i])<min(xs[i]-xs[i-1],xs[i+1]-xs[i])*.2]
            if candidates:
                choices=cluster(candidates)
                xs[i]=max(choices,key=lambda x:(sum(abs(v-x)<=tolerance for v in candidates),-abs(x-xs[i])))
        column_origin='model_columns'
    elif observed:
        xs=[float(left),*observed,float(right)];column_origin='visible_rules'
    else:raise ValueError('Column boundaries unresolved')
    if any(b-a<2 for a,b in zip(xs,xs[1:])):raise ValueError('Conflicting column boundaries')
    # A short mark inside a cell cannot authorize a new row. Separator ends
    # must reach actual column boundaries, including partial rules at spans.
    horizontal=[line for line in horizontal if flexible_rows or (
        min(abs(line[0]-x) for x in xs)<=3*tolerance
        and min(abs(line[2]-x) for x in xs)<=3*tolerance)]
    ys=cluster([line[1] for line in horizontal])
    if not 2<=len(ys)<=201:raise ValueError('Physical row boundaries unresolved')
    if not atomic and any(abs(line[1]-ys[0])<=3*tolerance and line[3]>ys[-1]+3*tolerance
           and min(abs(line[0]-left),abs(line[0]-right))<=x_tolerance for line in vertical):
        raise ValueError('Table bottom is clipped; preserve source for review')
    rows,columns=len(ys)-1,len(xs)-1
    if rows*columns>2048:raise ValueError('Physical grid exceeds cell budget')
    if atomic:
        return dict(rows=rows,columns=columns,x_cuts=xs,y_cuts=ys,column_origin=column_origin,cells=[
            dict(row=r+1,column=c+1,rowspan=1,colspan=1,bbox=[xs[c],ys[r],xs[c+1],ys[r+1]])
            for r in range(rows) for c in range(columns)])
    parents=list(range(rows*columns))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def join(a,b):parents[find(b)]=find(a)
    ruled=any(vertical_support(x)>=.5 for x in xs[1:-1])
    # The detector can express a colspan on a borderless axis. Align it to
    # physical rows and learned columns, as in TATR's structure postprocessing.
    model_groups=[]
    for a,b,c,d in spanning_boxes:
        rr=[r for r in range(rows) if b<=(ys[r]+ys[r+1])/2<=d]
        cc=[col for col in range(columns) if a<=(xs[col]+xs[col+1])/2<=c]
        if rr and len(cc)>1:model_groups.append((set(rr),set(cc)))
    for row in range(rows):
        for col in range(columns):
            identity=row*columns+col
            x0,x1=xs[col:col+2];y0,y1=ys[row:row+2]
            if row+1<rows:
                intervals=[(line[0],line[2]) for line in (row_segments or horizontal) if abs(line[1]-y1)<=tolerance]
                if coverage(intervals,x0+(x1-x0)*.12,x1-(x1-x0)*.12)<.5:
                    join(identity,identity+columns)
            # Missing dividers authorize a colspan only on an axis which has
            # actual internal ruling. Borderless model columns stay distinct.
            if col+1<columns:
                intervals=[(line[1],line[3]) for line in vertical if abs(line[0]-x1)<=x_tolerance]
                model_span=any(row in rr and col in cc and col+1 in cc for rr,cc in model_groups)
                if model_span or (ruled and coverage(intervals,y0+(y1-y0)*.03,y1-(y1-y0)*.03)<.9):
                    join(identity,identity+1)
    groups={}
    for row in range(rows):
        for col in range(columns):groups.setdefault(find(row*columns+col),[]).append((row,col))
    cells=[]
    for group in groups.values():
        r0=min(r for r,c in group);r1=max(r for r,c in group)+1
        c0=min(c for r,c in group);c1=max(c for r,c in group)+1
        if len(group)!=(r1-r0)*(c1-c0):raise ValueError('Ambiguous nonrectangular merged cell')
        cells.append(dict(row=r0+1,column=c0+1,rowspan=r1-r0,colspan=c1-c0,
            bbox=[xs[c0],ys[r0],xs[c1],ys[r1]]))
    return dict(rows=rows,columns=columns,x_cuts=xs,y_cuts=ys,column_origin=column_origin,
        cells=sorted(cells,key=lambda cell:(cell['row'],cell['column'])))


def table_rule_observations(image, column_boxes=None):
    """Measure image strokes; return evidence separately from reconstruction."""
    import cv2
    import numpy as np
    gray=cv2.cvtColor(np.array(image.convert('RGB')),cv2.COLOR_RGB2GRAY)
    height,width=gray.shape
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,15,2)
    horizontal_mask=cv2.morphologyEx(binary,cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT,(max(20,width//25),1)))
    # A shaded header has a real color boundary even where no dark rule is
    # drawn. Measure long horizontal contrast edges as separator evidence.
    contrast=cv2.convertScaleAbs(cv2.Sobel(gray,cv2.CV_16S,0,1,ksize=3))
    contrast=cv2.threshold(contrast,12,255,cv2.THRESH_BINARY)[1]
    contrast=cv2.morphologyEx(contrast,cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT,(max(20,width//25),1)))
    contrast=cv2.morphologyEx(contrast,cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT,(max(3,round(width*.008)),1)))
    horizontal_mask |= contrast
    vertical_mask=cv2.morphologyEx(binary,cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(18,height//15))))
    vcontrast=cv2.convertScaleAbs(cv2.Sobel(gray,cv2.CV_16S,1,0,ksize=3))
    vcontrast=cv2.threshold(vcontrast,12,255,cv2.THRESH_BINARY)[1]
    vcontrast=cv2.morphologyEx(vcontrast,cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(8,round(width*.004)))))
    vertical_mask |= vcontrast
    horizontal=[];horizontal_segments=[];vertical=[]
    for contour in cv2.findContours(horizontal_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,w,h=cv2.boundingRect(contour)
        segment=[x,y+(h-1)/2,x+w,y+(h-1)/2]
        if w>max(8,width*.02):horizontal_segments.append(segment)
        if w>width*.15:horizontal.append(segment)
    for contour in cv2.findContours(vertical_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,w,h=cv2.boundingRect(contour)
        if h>max(6,height*.01):vertical.append([x+(w-1)/2,y,x+(w-1)/2,y+h])
    return dict(width=width,height=height,horizontal=horizontal,horizontal_segments=horizontal_segments,vertical=vertical,
        ink=binary,horizontal_mask=horizontal_mask,vertical_mask=vertical_mask)


def infer_table_structure(image):
    """Read column and spanning-cell objects from the local structure model."""
    import torch
    from transformers import AutoImageProcessor,TableTransformerForObjectDetection
    location=os.getenv('TABLE_COLUMN_MODEL_PATH',str(Path(__file__).resolve().parents[2]/'scratch/a3-models/table-transformer'))
    processor=AutoImageProcessor.from_pretrained(location,local_files_only=True)
    model=TableTransformerForObjectDetection.from_pretrained(location,local_files_only=True).eval()
    try:
        with torch.inference_mode():prediction=model(**processor(images=image,return_tensors='pt'))
        result=processor.post_process_object_detection(prediction,threshold=.7,
            target_sizes=torch.tensor([[image.height,image.width]]))[0]
        objects=[dict(label=model.config.id2label[int(label)],bbox=box.tolist(),score=float(score))
            for label,box,score in zip(result['labels'],result['boxes'],result['scores'])]
        return dict(columns=[o['bbox'] for o in objects if o['label']=='table column'],
                    spans=[o['bbox'] for o in sorted(objects,key=lambda o:-o['score']) if o['label']=='table spanning cell'],
                    rows=[o['bbox'] for o in sorted(objects,key=lambda o:o['bbox'][1]) if o['label']=='table row'])
    finally:
        del model,processor


def infer_table_column_boxes(image):
    return infer_table_structure(image)['columns']


def corroborated_table_grid(image, rules, prediction, source_cells):
    """Require agreement for row topology; align model spans to a learned grid.

    Columns always come from the detector. Physical ruling can recover a row
    omitted by the model or coalesce wrapped OCR lines into physical cells.
    A disagreement is retained as unresolved, never accepted as a new fact.
    """
    import numpy as np
    columns=sorted(prediction['columns'],key=lambda b:b[0])
    model_rows=sorted(prediction.get('rows',[]),key=lambda b:b[1])
    if not columns or len(columns)>64:raise ValueError('Table model did not confirm columns')
    source_rows=max((c['row']+c.get('rowspan',1)-1 for c in source_cells),default=0)
    source_columns=max((c['column']+c.get('colspan',1)-1 for c in source_cells),default=0)
    if source_columns and source_columns!=len(columns) and not prediction.get('columns_confirmed'):
        raise ValueError('Independent column predictions disagree; retain original source')
    physical=None
    try:
        horizontal=rules['horizontal']
        native=prediction.get('structure_provider')=='surya_table_rec'
        if native and model_rows:
            # A letter's contrast edge is not a new physical row. Require a
            # nearby learned row boundary before using short/full ink rules.
            bands=[(b[1],b[3]) for b in model_rows]
            horizontal=[line for line in horizontal if any(
                min(abs(line[1]-a),abs(line[1]-b))<=max(3,(b-a)*.25) for a,b in bands)]
        physical=physical_table_grid(image.width,image.height,horizontal,rules['vertical'],columns,atomic=True,flexible_rows=native)
        # Padding can include a caption underline followed by an empty strip.
        # It is not a table row. Never trim a row containing source pixels.
        while physical['rows']>1:
            x0,x1=physical['x_cuts'][0],physical['x_cuts'][-1]
            y0,y1=physical['y_cuts'][:2];pad=min(max(3,round(min(image.size)*.006)),max(1,int((y1-y0)/4)))
            ink=rules['ink'][math.ceil(y0)+pad:math.floor(y1)-pad,math.ceil(x0)+pad:math.floor(x1)-pad]
            if ink.size==0 or np.count_nonzero(ink)>2:break
            physical['y_cuts'].pop(0);physical['rows']-=1
    except ValueError:pass
    physical_count=physical['rows'] if physical else 0
    full_rules=False
    if physical:
        width=physical['x_cuts'][-1]-physical['x_cuts'][0]
        full_rules=all(any(abs(l[1]-y)<4 and l[2]-l[0]>=width*.85 for l in rules['horizontal'])
                       for y in physical['y_cuts'])
    native_coalescing=(prediction.get('structure_provider')=='surya_table_rec' and physical
        and min(source_rows,prediction.get('reference_row_count',source_rows))<physical_count<max(source_rows,prediction.get('reference_row_count',source_rows))
        and physical['y_cuts'][0]>=model_rows[0][1]-5 and physical['y_cuts'][-1]<=model_rows[-1][3]+5)
    if prediction.get('structure_provider') != 'surya_table_rec' and model_rows:
        # The learned TATR row boxes define rows even on borderless tables.
        # Missing interior rules do not prove that separate rows are merged.
        origin='tatr_model_rows';use_physical=False
    elif native_coalescing:
        origin='physical_cells_with_native_columns';use_physical=True
    elif physical and full_rules and len(model_rows)>=2*physical_count:
        origin='physical_cells_override_wrapped_model_rows';use_physical=True
    elif source_rows and source_rows==len(model_rows):
        origin='source_model_row_consensus';use_physical=False
    elif physical and source_rows==physical_count:
        origin='physical_source_row_consensus';use_physical=True
    elif physical and physical_count==len(model_rows):
        origin='physical_model_row_consensus';use_physical=False
    elif physical and source_rows and min(source_rows,len(model_rows))<physical_count<max(source_rows,len(model_rows)):
        origin='physical_rows_between_independent_predictions';use_physical=True
    else:raise ValueError('Independent row predictions disagree; retain original source')
    native=prediction.get('structure_provider')=='surya_table_rec'
    if native and physical and physical_count==len(model_rows):
        use_physical=True;origin='physical_native_row_consensus'
    if use_physical and not native:
        # Re-run on the selected boundaries so source/caption padding cannot
        # reintroduce a trimmed row. Geometry, rather than text, owns spans.
        selected=[l for l in rules['horizontal'] if physical['y_cuts'][0]-2<=l[1]<=physical['y_cuts'][-1]+2]
        grid=physical_table_grid(image.width,image.height,selected,rules['vertical'],columns,row_segments=rules.get('horizontal_segments'))
        grid['row_origin']=origin
        return grid
    xs=[max(0,min(b[0] for b in columns)),*[(a[2]+b[0])/2 for a,b in zip(columns,columns[1:])],
        min(image.width,max(b[2] for b in columns))]
    ys=[max(0,model_rows[0][1]),*[(a[3]+b[1])/2 for a,b in zip(model_rows,model_rows[1:])],
        min(image.height,model_rows[-1][3])]
    if physical:
        xs=physical['x_cuts'][:]
    if use_physical:
        ys=physical['y_cuts'][:]
    elif native and physical and len(physical['y_cuts'])>=len(ys):
        # Match learned boundaries to distinct observed rules. A glyph edge
        # can add an extra rule inside a header; it must not shift all rows.
        candidates=physical['y_cuts'];states={0:(0.,[])}
        for target in ys:
            following={};best=None
            for j,value in enumerate(candidates):
                if j in states and (best is None or states[j][0]<best[0]):best=states[j]
                if best is not None:following[j+1]=(best[0]+abs(value-target),best[1]+[value])
            states=following
        if states:
            matched=min(states.values(),key=lambda item:item[0])[1]
            limits=[max(8,(ys[min(i+1,len(ys)-1)]-ys[max(0,i-1)])*.45) for i in range(len(ys))]
            if all(abs(a-b)<=limit for a,b,limit in zip(ys,matched,limits)):
                ys=matched;origin='native_rows_snapped_to_observed_rules'
    rows,cols=len(ys)-1,len(xs)-1
    if rows*cols>2048 or any(b<=a for cuts in (xs,ys) for a,b in zip(cuts,cuts[1:])):
        raise ValueError('Invalid learned table grid')
    row_xs=[xs[:] for _ in range(rows)]
    if native:
        for row in range(rows):
            y0,y1=ys[row:row+2]
            for col in range(cols-1):
                low=(xs[col]+xs[col+1])/2;high=(xs[col+1]+xs[col+2])/2
                candidates=[l for l in rules['vertical'] if low<l[0]<high
                    and l[1]<=y0+(y1-y0)*.1 and l[3]>=y1-(y1-y0)*.1]
                if candidates:
                    row_xs[row][col+1]=min(candidates,key=lambda l:abs(l[0]-xs[col+1]))[0]
    def has_divider(row,col):
        y0,y1=ys[row:row+2];x=row_xs[row][col+1]
        radius=max(3,min(xs[col+1]-xs[col],xs[col+2]-xs[col+1])*.05) if native else min(xs[col+1]-xs[col],xs[col+2]-xs[col+1])*.2
        inset=.15 if prediction.get('structure_provider')=='surya_table_rec' else .03
        return any(abs(l[0]-x)<=radius and l[1]<=y0+(y1-y0)*inset
                   and l[3]>=y1-(y1-y0)*inset for l in rules['vertical'])
    def row_separators(row):
        radius=min(ys[row+1]-ys[row],ys[row+2]-ys[row+1])*.12
        return [l for l in rules.get('horizontal_segments',rules['horizontal']) if abs(l[1]-ys[row+1])<=max(3,radius)]
    def row_divides(row,col):
        a,b=row_xs[row][col:col+2]
        return any(min(b,l[2])-max(a,l[0])>=(b-a)*.5 for l in row_separators(row))
    groups=[{(r,c)} for r in range(rows) for c in range(cols)]
    for a,b,c,d in prediction.get('spans',[]):
        rr=[r for r in range(rows) if b<=(ys[r]+ys[r+1])/2<=d]
        cc=[k for k in range(cols) if a<=(xs[k]+xs[k+1])/2<=c]
        region={(r,k) for r in rr for k in cc}
        if len(region)<2:continue
        if any(has_divider(r,k) for r in rr for k in cc if k+1 in cc):continue
        if any(row_divides(r,k) for r in rr if r+1 in rr for k in cc):continue
        overlaps=[g for g in groups if g & region]
        if any(not g<=region for g in overlaps):continue  # TATR span NMS.
        groups=[g for g in groups if not g & region]+[region]
    # A missing line is not evidence of a merged cell: borderless tables and
    # partially ruled tables still have separate model-predicted cells.
    cells=[]
    for group in groups:
        r0=min(r for r,c in group);r1=max(r for r,c in group)+1
        c0=min(c for r,c in group);c1=max(c for r,c in group)+1
        cells.append(dict(row=r0+1,column=c0+1,rowspan=r1-r0,colspan=c1-c0,
            bbox=[min(row_xs[r][c0] for r in range(r0,r1)),ys[r0],max(row_xs[r][c1] for r in range(r0,r1)),ys[r1]]))
    return dict(rows=rows,columns=cols,x_cuts=xs,y_cuts=ys,row_origin=origin,column_origin='model_columns',
                cells=sorted(cells,key=lambda c:(c['row'],c['column'])))


def native_table_structure(image, predictor, original, source_cells):
    """Use Surya's trained table_rec rows/columns when the first grid fails."""
    import numpy as np
    from surya.table_rec import TableRecPredictor
    framed=frame_text_crop(image)
    result=TableRecPredictor(manager=predictor.manager)([framed])[0]
    if result.error:raise ValueError('Surya table structure prediction failed')
    scale=1 if framed is image else min(4,max(1,64/image.height),2048/image.width)
    dx=(framed.width-round(image.width*scale))//2;dy=(framed.height-round(image.height*scale))//2
    def original_box(box):
        return [max(0,min(image.width,(box[0]-dx)/scale)),max(0,min(image.height,(box[1]-dy)/scale)),
                max(0,min(image.width,(box[2]-dx)/scale)),max(0,min(image.height,(box[3]-dy)/scale))]
    rows=sorted([original_box(r.bbox) for r in result.rows],key=lambda b:b[1])
    columns=sorted([original_box(c.bbox) for c in result.cols],key=lambda b:b[0])
    rows=[b for b in rows if b[3]>b[1]];columns=[b for b in columns if b[2]>b[0]]
    if not rows or not columns:raise ValueError('Surya did not confirm table rows and columns')
    rules=table_rule_observations(image)
    # Padding is often returned as a row by tiny-table inference. Remove it
    # only when its interior contains no text pixels after separator removal.
    def blank(box):
        a,b,c,d=map(lambda v:max(0,int(v)),box)
        ink=rules['ink'][b+3:min(image.height,d)-3,a+3:min(image.width,c)-3].copy()
        if not ink.size:return True
        separators=rules['horizontal_mask'][b+3:min(image.height,d)-3,a+3:min(image.width,c)-3].copy()
        if 'vertical_mask' in rules:separators |= rules['vertical_mask'][b+3:min(image.height,d)-3,a+3:min(image.width,c)-3]
        ink[separators>0]=0
        return np.count_nonzero(ink)<=2
    while len(rows)>1 and blank(rows[0]):rows.pop(0)
    while len(rows)>1 and blank(rows[-1]):rows.pop()
    source_rows=max((c['row']+c.get('rowspan',1)-1 for c in source_cells),default=0)
    source_columns=max((c['column']+c.get('colspan',1)-1 for c in source_cells),default=0)
    primary_rows=original.get('rows',[])
    # A neighboring caption can be inside the padded crop but above a table
    # independently located by the source and first structure detector.
    if primary_rows and len(original['columns'])==source_columns:
        top=min((b[1] for b in primary_rows),default=0)
        while len(rows)>source_rows and rows[0][3]<=top:rows.pop(0)
    if source_rows==1 and len(rows)>1:
        # A narrow header can be split into padding/text bands. The full-HTML
        # task supplies a second learned row-count observation for this case.
        full=TableRecPredictor(manager=predictor.manager).predict_full([framed])[0]
        if not full.error:
            cells=source_html_metadata(full.html or full.raw or '').get('table_cells',[])
            populated={c['row'] for c in cells if c.get('text','').strip()}
            if len(populated)==1:
                rows=[[min(b[0] for b in rows),min(b[1] for b in rows),max(b[2] for b in rows),max(b[3] for b in rows)]]
    confirmed=len(columns) in {len(original['columns']),source_columns}
    if not confirmed:raise ValueError('Native columns lack independent confirmation')
    primary_columns=sorted(original['columns'],key=lambda b:b[0])
    if len(columns)==len(primary_columns):
        extent=primary_columns[-1][2]-primary_columns[0][0]
        coverage=sum(max(0,b[2]-b[0]) for b in primary_columns)/max(1,extent)
        if coverage>=.8:columns=primary_columns
    return dict(rows=rows,columns=columns,spans=original.get('spans',[]) if len(columns)==len(original['columns']) else [],columns_confirmed=True,reference_row_count=len(primary_rows),
                structure_provider='surya_table_rec')


def recognize_physical_table(image, predictor, source_cells=(), *, column_predictor=None):
    if column_predictor is not None:
        return _recognize_physical_table(image,predictor,source_cells,column_predictor=column_predictor)
    prediction=infer_table_structure(image)
    try:
        return _recognize_physical_table(image,predictor,source_cells,column_predictor=lambda im:prediction)
    except ValueError as initial:
        recovered=native_table_structure(image,predictor,prediction,source_cells)
        try:
            result=_recognize_physical_table(image,predictor,source_cells,
                column_predictor=lambda im:recovered,force_cell_ocr=True)
        except ValueError as error:
            raise ValueError(f'{error}; initial grid: {initial}') from error
        result['table_structure_review'].update(recovery='surya_table_rec',initial_grid_reason=str(initial))
        return result


def table_crop_has_text(ink):
    """Ignore long separators; preserve short isolated glyphs such as I/1."""
    import numpy as np
    if not ink.size:return False
    height,width=ink.shape
    clean=ink.copy()
    horizontal=(np.count_nonzero(ink,axis=1)>=width*.85)&np.any(ink[:,:2],axis=1)&np.any(ink[:,-2:],axis=1)
    vertical=(np.count_nonzero(ink,axis=0)>=height*.85)&np.any(ink[:2,:],axis=0)&np.any(ink[-2:,:],axis=0)
    clean[horizontal,:]=0
    clean[:,vertical]=0
    return np.count_nonzero(clean[2:-2,2:-2] if min(clean.shape)>4 else clean)>2


def normalize_table_ocr_comparison(text):
    """Compare equivalent OCR markup without changing extracted text."""
    import re
    import unicodedata
    if '\\' in text:
        from latex2mathml.symbols_parser import convert_symbol
        # A TeX command followed by _4 or _{4} renders a subscript. Preserve
        # the characters while discarding only this markup for comparison.
        text=re.sub(r'(\\[A-Za-z]+)_\s*\{([^{}]*)\}',r'\1\2',text)
        text=re.sub(r'(\\[A-Za-z]+)_([0-9])',r'\1\2',text)
        def symbol(match):
            code=convert_symbol(match[0])
            try:return chr(int(code,16)) if code else match[0]
            except ValueError:return match[0]
        text=re.sub(r'\\[A-Za-z]+',symbol,text)
    text=re.sub(r'\\(?:mathrm|textrm|text|mathbf|mathsf)\s*\{([^{}]*)\}',r'\1',text)
    text=re.sub(r'\^\s*\{([^{}]*)\}',r'\1',text)
    text=re.sub(r'\^([\u2218\d])',r'\1',text)
    text=unicodedata.normalize('NFKC',text).translate(str.maketrans({
        '\u2022':'','\u00ae':'','\u2122':'','*':'','\u2212':'-','\u2013':'-','\u2014':'-',
        '\u00b0':'\u2218','\u00d7':'x','\u00d8':'\u03c6','\u2300':'\u03c6','\u2205':'\u03c6'}))
    # OCR can insert an apostrophe between a label and its numeric footnote.
    # This affects comparison only; keep apostrophes inside words and numbers.
    text=re.sub(r"(?<=[^\W\d_])['\u2019](?=\d)", '', text)
    return ''.join(text.split()).casefold()


def table_structure_warnings(cells, source_cells, reconstructed_columns):
    """Expose conflicting source columns without changing the model output."""
    normalize=normalize_table_ocr_comparison
    source_columns=max((c.get('column',0)+c.get('colspan',1)-1 for c in source_cells),default=0)
    warnings=[]
    if source_columns and source_columns!=reconstructed_columns:
        warnings.append('source_column_count_differs_from_reconstructed_grid')
    quotes=[(c.get('row'),c.get('column'),normalize(c.get('text',''))) for c in source_cells]
    for cell in cells:
        value=normalize(cell.get('text',''))
        matches=[]
        for row,column,quote in quotes:
            if not quote:continue
            start=value.find(quote)
            if start>=0:matches.append((row,column,start,start+len(quote)))
        if any(a[0]==b[0] and a[1]!=b[1] and (a[3]<=b[2] or b[3]<=a[2])
               for i,a in enumerate(matches) for b in matches[i+1:]):
            warnings.append('multiple_source_columns_combined_in_single_cell')
            break
    return warnings


def _recognize_physical_table(image, predictor, source_cells=(), *, column_predictor=None,force_cell_ocr=False):
    """Read the physical cells using Surya's table-specific OCR."""
    import numpy as np
    from PIL import Image,ImageOps
    from surya.inference.schema import BatchInputItem
    prediction=column_predictor(image) if column_predictor else infer_table_structure(image)
    columns=prediction['columns'] if isinstance(prediction,dict) else prediction
    spans=prediction.get('spans',[]) if isinstance(prediction,dict) else []
    rules=table_rule_observations(image,columns)
    if not columns:raise ValueError('Table model did not confirm column boundaries')
    if isinstance(prediction,dict):
        grid=corroborated_table_grid(image,rules,prediction,source_cells)
    else:
        grid=physical_table_grid(image.width,image.height,rules['horizontal'],rules['vertical'],columns,spans)
    normalized=normalize_table_ocr_comparison
    def table_frame(crop,*,rescue=False):
        if rescue:
            # The final table-only retry gives short lines more pixel height;
            # figure OCR's framing policy remains separate.
            scale=min(3,max(1,128/crop.height),2048/crop.width)
            if scale>1:crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
        return frame_text_crop(crop,rescue=rescue)
    header_roles={}
    source_columns=max((c.get('column',0)+c.get('colspan',1)-1 for c in source_cells),default=0)
    for cell in source_cells:
        if cell.get('text','').strip() and type(cell.get('header')) is bool:
            header_roles.setdefault(normalized(cell['text']),set()).add(cell['header'])
    batch=[];pending=[];cell_boxes={};occupied={}
    # Each cell consumes GPU time, so a fixed 90-second budget can expire
    # before later cells even enter inference on a large table.
    deadline=time.perf_counter()+max(240,30*len(grid['cells']))
    for index,cell in enumerate(grid['cells']):
        a,b,c,d=cell['bbox']
        if a>=c or b>=d:raise ValueError('Physical cell has invalid bounds')
        box=(max(0,math.floor(a)),max(0,math.floor(b)),
             min(image.width,math.ceil(c)),min(image.height,math.ceil(d)))
        if box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Physical cell has invalid bounds')
        cell.update(text='',header=False,ocr_status='empty_source_crop')
        ink=rules['ink'][box[1]:box[3],box[0]:box[2]].copy()
        # Do not erase every detected vertical stroke here: glyphs such as
        # I/1 also contain long strokes. Keep the entire predicted cell for OCR.
        # Retain the previous border exclusion only for the existing ink
        # check; never remove those pixels from the image sent to OCR.
        margin=max(2,round(min(image.width,image.height)*.004))+3
        interior=ink[margin:-margin,margin:-margin] if min(ink.shape)>2*margin else ink
        if np.count_nonzero(interior)<=2:continue
        if force_cell_ocr and not table_crop_has_text(ink):
            cell['ocr_status']='empty_separator_only_crop'
            continue
        occupied[index]=cell
        cell_boxes[index]=box
    # Read each detected nonempty cell with the table-specific prompt.
    # Page OCR may inform structure, never the cell text itself.
    for index,cell in occupied.items():
        crop=ImageOps.expand(image.crop(cell_boxes[index]).convert('RGB'),border=8,fill='white')
        batch.append(BatchInputItem(image=frame_text_crop(crop),prompt_type='block',max_tokens=1024,
            metadata=dict(block_idx=index,table_cell_ocr=True,table_cell_deadline=deadline)))
        pending.append(cell)
    results=[]
    for offset in range(0,len(batch),128):
        results.extend(predictor.manager.generate(batch[offset:offset+128]))
    if len(results)!=len(pending):raise ValueError('Surya physical-cell OCR count mismatch')
    for cell,result in zip(pending,results):
        text=_html_text(result.raw or '')
        if result.error or not text.strip():
            box=cell_boxes[next(i for i,c in occupied.items() if c is cell)]
            crop=ImageOps.expand(image.crop(box).convert('RGB'),border=8,fill='white')
            check=predictor.manager.generate([BatchInputItem(image=table_frame(crop,rescue=True),
                prompt_type='block',max_tokens=1024,
                metadata=dict(table_cell_ocr=True,table_cell_deadline=deadline))])
            second=_html_text(check[0].raw or '') if len(check)==1 and not check[0].error else None
            if second is None:
                cell.update(text='', ocr_status='ocr_failed')
                continue
            if not second.strip() and (not result.error or not table_crop_has_text(rules['ink'][box[1]:box[3],box[0]:box[2]])):
                cell.update(text='',ocr_status='confirmed_empty_crop')
                continue
            if not second.strip():
                cell.update(text='', ocr_status='unverified')
                continue
            text=second
        cell.update(text=text,ocr_status='completed')
        roles=header_roles.get(normalized(text),set())
        cell['header']=roles=={True}
    grid['cells'].sort(key=lambda c:(c['row'],c['column']))
    rows=[['']*grid['columns'] for _ in range(grid['rows'])]
    for cell in grid['cells']:
        rows[cell['row']-1][cell['column']-1]=cell['text']
    structure_warnings=[]
    if any(c.get('ocr_status') in {'ocr_failed','unverified'} for c in grid['cells']):
        structure_warnings.append('cell_ocr_incomplete_or_unverified')
    return dict(table_rows=rows,table_cells=grid['cells'],table_structure_status='cell_aware',
        table_header_origin='surya_cell_source_headers_unreviewed',table_structure_policy=TABLE_CELL_POLICY,
        table_structure_review=dict(policy=TABLE_CELL_POLICY,row_origin=grid.get('row_origin','visible_rules'),
            column_origin=grid['column_origin'],rows=grid['rows'],columns=grid['columns'],
            x_cuts=grid['x_cuts'],y_cuts=grid['y_cuts'],bbox_scope='table_crop_pixels',
            image_size=list(image.size),ocr_provider='surya-ocr-v2',line_breaks_create_rows=False,
            source_column_count=source_columns,
            structure_warnings=structure_warnings,
            text_source='surya_table_cell_ocr'))


def native_reading_order(blocks, scope='page', source='surya_layout_model'):
    """Use Surya's trained output order, never OCR completion time or bbox sorting.

    Surya 2 exposes the layout sequence as reading_order (box.position).
    Validate the sequence structurally; this is not a visual accuracy score.
    Missing/duplicate ranks retain evidence but cannot authorize NEXT edges.
    """
    blocks = list(blocks)
    ranks = [getattr(block, 'reading_order', None) for block in blocks]
    valid = all(type(rank) is int and rank >= 0 for rank in ranks)
    valid = valid and len(set(ranks)) == len(ranks)
    ordered = sorted(zip(blocks, ranks), key=lambda item: item[1]) if valid else zip(blocks, ranks)
    return [(block, dict(reading_order=rank if type(rank) is int and rank >= 0 else None,
                         reading_order_source=source, reading_order_scope=scope,
                         reading_order_status='model_predicted' if valid else 'unresolved',
                         reading_order_unresolved=not valid)) for block, rank in ordered]


def _bind_windows_child_lifetime(process):
    """Close a Windows job on parent exit, including a native crash."""
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                    ('PerJobUserTimeLimit', ctypes.c_longlong),
                    ('LimitFlags', wintypes.DWORD),
                    ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t),
                    ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t),
                    ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
            'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IoCounters),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.get_last_error(); kernel.CloseHandle(job); raise ctypes.WinError(error)
    if not kernel.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle))):
        error = ctypes.get_last_error(); kernel.CloseHandle(job); raise ctypes.WinError(error)
    return job


def _close_windows_child_job(job):
    if job is not None:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        if not kernel.CloseHandle(job):
            raise ctypes.WinError(ctypes.get_last_error())


class ManagedSuryaBackend:
    """Own the existing local binary, restart crashes without downloading weights."""

    def __init__(self, backend):
        self.backend, self.process, self.job = backend, None, None
        self.lock = RLock()
        self.url = os.environ['SURYA_INFERENCE_URL'].rstrip('/')
        parsed = urlparse(self.url)
        if parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.scheme != 'http':
            raise ValueError('Managed Surya requires a local HTTP endpoint')
        self.port = parsed.port
        atexit.register(self.suspend)

    def ready(self):
        try:
            with urlopen(self.url+'/models', timeout=2) as response:
                return any(row['id'] == 'datalab-to/surya-ocr-2'
                           for row in json.load(response).get('data', []))
        except (OSError, ValueError, KeyError):
            return False

    def start(self):
        with self.lock:
            if not self.ready():
                self.suspend()
                binary = Path(os.environ['LLAMA_CPP_BINARY'])
                model = Path(os.environ['SURYA_GGUF_LOCAL_MODEL_PATH'])
                projection = Path(os.environ['SURYA_GGUF_LOCAL_MMPROJ_PATH'])
                template = model.parent/'chat_template.jinja'
                if not all(path.is_file() for path in (binary, model, projection, template)):
                    raise RuntimeError('Existing local Surya files are unavailable')
                # Never start over an unrelated service bound to this port.
                import socket
                with socket.socket() as check:
                    if check.connect_ex(('127.0.0.1', self.port)) == 0:
                        raise RuntimeError('Surya port is occupied by an unhealthy or different service')
                command = [str(binary), '-m', str(model), '--mmproj', str(projection),
                    '--chat-template-file', str(template), '--image-min-tokens', '1024',
                    '--alias', 'datalab-to/surya-ocr-2', '--host', '127.0.0.1', '--port', str(self.port),
                    '--parallel', os.getenv('SURYA_INFERENCE_PARALLEL', '2'),
                    '--ctx-size', os.getenv('SURYA_INFERENCE_CTX_SIZE', '32768'),
                    '--n-gpu-layers', '99', '--jinja']
                log = model.parent.parent.parent/'surya-managed.log'
                with log.open('ab') as output:
                    self.process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    self.job = _bind_windows_child_lifetime(self.process)
                except Exception:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                    self.process = None
                    raise
                print(json.dumps(dict(event='surya_backend', status='starting', pid=self.process.pid)), flush=True)
                deadline = time.monotonic()+60
                while not self.ready():
                    if self.process.poll() is not None or time.monotonic() >= deadline:
                        self.suspend()
                        raise RuntimeError('Surya backend could not recover within startup budget')
                    time.sleep(.25)
                self.backend.stop()  # Invalidate the library's stale HTTP handle.
                print(json.dumps(dict(event='surya_backend', status='ready', pid=self.process.pid)), flush=True)
            return self.backend.start()

    def capacity(self):
        return self.backend.capacity()

    def suspend(self):
        with self.lock:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
                self.process = None
                job, self.job = self.job, None
                _close_windows_child_job(job)
                self.backend.stop()
                print(json.dumps(dict(event='surya_backend', status='released')), flush=True)


def release_surya_backend():
    """Release only the process owned by this runtime, for CUDA recovery."""
    if _managed_backend is not None:
        _managed_backend.suspend()


def repeating_output(text):
    """Detect repeated content, not indentation or a drawing's separator line."""
    tail = ' '.join(text[-4000:].split())
    for cut in (0, 50):
        value = tail[:-cut] if cut else tail
        for size in range(24, min(500, len(value)//4)+1):
            unit = value[-size:]
            if any(char.isalnum() for char in unit) and value.endswith(unit*4):
                return True
    return False


def frame_text_crop(image, rescue=False):
    """Keep every source pixel; give tiny/narrow text stable image-grid margins."""
    from PIL import Image, ImageOps
    if not rescue and image.height>=48 and image.width<=12*image.height:
        return image
    crop=image.convert('RGB')
    if rescue:crop=ImageOps.autocontrast(crop)
    scale=min(4,max(1,64/crop.height),2048/crop.width)
    if scale>1:
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
    frame=Image.new('RGB',(max(256,crop.width+32),max(128,crop.height+32)),'white')
    frame.paste(crop,((frame.width-crop.width)//2,(frame.height-crop.height)//2))
    return frame


class SuryaStreamManager:
    """Use Surya's prompts/parsers, cancel repetition before decoding a whole page again."""

    def __init__(self, manager):
        self.manager = manager
        self.client = None
        self.begin_page()

    def begin_page(self):
        self.started = time.perf_counter()
        self.calls = []

    def generate(self, batch, *, retry=False, retry_reason=None, rescue=False):
        if not batch:
            return []
        from openai import OpenAI
        from surya.inference.prompts import PROMPT_MAPPING
        from surya.inference.schema import BatchOutputItem
        from surya.inference.util import scale_to_fit
        from surya.settings import settings
        handle = self.manager.backend.start()
        if self.client is None:
            # Disable hidden HTTP retries as well as the library's three OCR retries.
            self.client = OpenAI(api_key='EMPTY', base_url=handle.base_url, max_retries=0)
        def read(item):
            started = time.perf_counter()
            page_budget = float(os.getenv('SURYA_PAGE_GENERATION_TIMEOUT', '0'))
            request_budget = float(os.getenv('SURYA_REQUEST_GENERATION_TIMEOUT', '0'))
            remaining = page_budget-(started-self.started) if page_budget>0 else float('inf')
            limit = min(remaining,request_budget if request_budget>0 else float('inf'))
            if item.metadata.get('table_cell_ocr'):
                limit=min(limit,45,item.metadata['table_cell_deadline']-started)
            visual_block=item.prompt_type=='block' and (item.metadata.get('visual_ocr') or getattr(self,'visual_phase',False))
            crop=frame_text_crop(item.image,rescue=True) if visual_block and rescue else item.image
            buffer = io.BytesIO()
            submitted_image=scale_to_fit(crop)
            submitted_image.save(buffer, format='PNG')
            messages = [dict(role='user', content=[dict(type='image_url', image_url=dict(
                url='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'))),
                dict(type='text', text=item.prompt or PROMPT_MAPPING[item.prompt_type])])]
            token_budget = item.max_tokens or 2048
            if item.prompt_type == 'block':
                # Layout count estimates omit markup/code punctuation. Provide headroom
                # up front; EOS still ends short crops immediately without a retry.
                minimum=max(1024,int(os.getenv('SURYA_MIN_BLOCK_TOKENS','1024')))
                if retry:minimum*=2
                token_budget = min(8192, max(minimum, token_budget*2))
                if rescue:token_budget=min(token_budget,1024)
            options = dict(model=handle.model_name, messages=messages, stream=True,
                stream_options={'include_usage': True}, temperature=0,
                top_p=.1,
                max_tokens=token_budget, timeout=max(1, min(limit, 30)),
                logprobs=bool(item.request_logprobs or settings.SURYA_INFERENCE_LOGPROBS))
            if item.guided_json is not None:
                options['response_format'] = dict(type='json_schema', json_schema=dict(
                    name='surya_layout', schema=item.guided_json, strict=True))
            if item.guided_regex is not None:
                options['extra_body'] = {'guided_regex': item.guided_regex}
            raw, status, tokens, logprobs, last_check = '', 'unfinished', 0, [], 0
            try:
                if limit <= 0:
                    raise TimeoutError('Surya page generation budget exceeded')
                with self.client.chat.completions.create(**options) as stream:
                    for chunk in stream:
                        if time.perf_counter()-started >= limit:
                            status = 'deadline_exceeded'
                            break
                        if chunk.usage:
                            tokens = chunk.usage.completion_tokens
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        raw += choice.delta.content or ''
                        if choice.logprobs and choice.logprobs.content:
                            logprobs.extend(entry.model_dump() for entry in choice.logprobs.content)
                        whitespace_run = len(raw)-len(raw.rstrip(' \t\u00a0'))
                        if whitespace_run >= max(256, item.image.width//2):
                            status = 'whitespace_loop_cancelled'
                            break
                        if len(raw) >= (384 if rescue else 1000) and len(raw)-last_check >= (128 if rescue else 200):
                            last_check = len(raw)
                            if repeating_output(raw):
                                status = 'repetition_cancelled'
                                break
                        if choice.finish_reason:
                            status = 'completed' if choice.finish_reason == 'stop' else 'token_budget_exceeded'
                # Closing the SSE stream cancels the llama.cpp task, including loops.
            except Exception as error:
                status = type(error).__name__
            probabilities = [math.exp(entry['logprob']) for entry in logprobs if entry.get('logprob') is not None]
            mean = sum(probabilities)/len(probabilities) if probabilities else None
            event = dict(prompt=item.prompt_type, status=status, elapsed_s=round(time.perf_counter()-started, 3),
                         output_tokens=tokens or None, output_chars=len(raw), retries=int(retry),
                         block_index=item.metadata.get('block_idx'), max_new_tokens=token_budget,
                         retry_reason=retry_reason, rescue=rescue, region_index=item.metadata.get('page_idx'),
                         source_image_size=list(item.image.size), input_image_size=list(submitted_image.size))
            if item.metadata.get('visual_ocr') or getattr(self,'visual_phase',False):
                event['visual_ocr'] = True
            self.calls.append(event)
            print(json.dumps(dict(event='surya_runtime', **event)), flush=True)
            if (not retry and item.prompt_type == 'block' and
                    status in {'whitespace_loop_cancelled', 'token_budget_exceeded',
                               'APITimeoutError', 'APIConnectionError', 'ReadError', 'RemoteProtocolError'} and
                    (page_budget<=0 or time.perf_counter()-self.started < page_budget)):
                # Retry only this failed crop once; optional deadlines are opt-in.
                replacement = copy(item)
                if status == 'whitespace_loop_cancelled':
                    from PIL import Image
                    # Wider framing prevents the decoder's endless <pre> indentation.
                    # Every original pixel and the source layout bbox remain unchanged.
                    framed = Image.new('RGB', (int(item.image.width*1.32), item.image.height), 'white')
                    framed.paste(item.image.convert('RGB'), (0, 0))
                    replacement.image = framed
                # Transport failure does not justify changing pixels, sampling or
                # token budget. Retry the original crop; completed siblings stay.
                if status not in {'APITimeoutError', 'APIConnectionError', 'ReadError', 'RemoteProtocolError'}:
                    replacement.max_tokens = min(4096, (item.max_tokens or 2048)*2)
                return self.generate([replacement], retry=True, retry_reason=status)[0]
            if (visual_block and retry and not rescue and
                    status == 'whitespace_loop_cancelled' and
                    (page_budget<=0 or time.perf_counter()-self.started<page_budget)):
                # One final, deterministic rescue; never repeat successful siblings
                # or recurse through pictures. Failed output remains unavailable.
                replacement=copy(item)
                replacement.temperature,replacement.top_p=0,.1
                return self.generate([replacement],retry=True,retry_reason=status,rescue=True)[0]
            return BatchOutputItem(raw=raw if status == 'completed' else '', token_count=tokens,
                error=status != 'completed', mean_token_prob=mean, logprobs=logprobs,
                metadata=item.metadata)
        workers = min(len(batch), max(1, int(os.getenv('SURYA_CROP_CONCURRENCY', '2'))),
                      self.manager.backend.capacity())
        if workers <= 1:
            return [read(item) for item in batch]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='surya-crop') as pool:
            # map retains source order even when a later crop finishes first.
            return list(pool.map(read, batch))


def create_surya_predictor():
    """One Surya backend, with bounded streaming over layout and OCR crops."""
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor
    global _managed_backend
    manager = SuryaInferenceManager()
    if os.getenv('SURYA_MANAGED_BACKEND') == '1':
        if _managed_backend is None:
            _managed_backend = ManagedSuryaBackend(manager.backend)
        manager.backend = _managed_backend
    return RecognitionPredictor(SuryaStreamManager(manager))

def _html_text(html):
    from html.parser import HTMLParser

    class TextParser(HTMLParser):

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []

        def handle_starttag(self, tag, attrs):
            if tag in {'br', 'p', 'div', 'li', 'tr', 'h1', 'h2', 'h3', 'h4'}:
                self.parts.append('\n')

        def handle_endtag(self, tag):
            if tag in {'td', 'th'}:
                self.parts.append('\t')
            elif tag in {'p', 'div', 'li', 'tr', 'h1', 'h2', 'h3', 'h4'}:
                self.parts.append('\n')

        def handle_data(self, data):
            self.parts.append(data)
    parser = TextParser()
    parser.feed(html)
    parser.close()
    return '\n'.join(line.strip() for line in ''.join(parser.parts).splitlines() if line.strip())


def source_html_metadata(html, *, strict_table=False):
    """Preserve Surya's explicit HTML structure; never infer levels or columns."""
    from html.parser import HTMLParser

    class StructureParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.rows, self.cells, self.headings = [], [], []
            self.row, self.cell, self.heading, self.table_depth = None, None, None, 0
            self.header_depth = 0
            self.invalid_table = False
            self.table_count = 0
            self.nested_table = False

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'table':
                if self.table_depth == 0: self.table_count += 1
                else: self.nested_table = True
                self.table_depth += 1
            if tag == 'thead' and self.table_depth == 1: self.header_depth += 1
            if tag in {'h1','h2','h3','h4','h5','h6'}:
                self.heading = dict(level=int(tag[1]),parts=[])
            if self.table_depth == 1 and tag == 'tr': self.row = []
            if self.table_depth == 1 and self.row is not None and tag in {'td','th'}:
                try:
                    rowspan, colspan = int(attrs.get('rowspan',1)), int(attrs.get('colspan',1))
                    if not 1 <= rowspan <= 200 or not 1 <= colspan <= 100:
                        self.invalid_table = True; self.cell = None; return
                except (TypeError,ValueError):
                    self.invalid_table = True; self.cell = None; return
                self.cell = dict(parts=[],rowspan=rowspan,colspan=colspan,header=tag=='th' or self.header_depth>0)
            if tag in {'br','p','div','li'} and self.cell is not None: self.handle_data('\n')

        def handle_data(self, data):
            if self.cell is not None: self.cell['parts'].append(data)
            if self.heading is not None: self.heading['parts'].append(data)

        def handle_endtag(self, tag):
            if tag in {'p','div','li'} and self.cell is not None: self.handle_data('\n')
            if tag in {'h1','h2','h3','h4','h5','h6'} and self.heading is not None:
                self.headings.append(self.heading); self.heading = None
            if tag in {'td','th'} and self.cell is not None:
                self.row.append(self.cell); self.cell = None
            if tag == 'tr' and self.row is not None:
                self.rows.append(self.row); self.row = None
            if tag == 'table': self.table_depth = max(0,self.table_depth-1)
            if tag == 'thead': self.header_depth = max(0,self.header_depth-1)

    parser = StructureParser()
    parser.feed(html or ''); parser.close()
    metadata = {}
    if strict_table and (parser.table_count!=1 or parser.table_depth or parser.nested_table
            or parser.cell is not None or parser.row is not None):
        return dict(table_structure_status='unresolved',table_structure_error='incomplete_or_multiple_tables')
    if len(parser.headings) == 1 and ''.join(parser.headings[0]['parts']).strip() == _html_text(html).strip():
        metadata.update(heading_level=parser.headings[0]['level'],heading_source='surya_html',
                        hierarchy_level_status='model_supplied')
    if parser.invalid_table:
        metadata.update(table_structure_status='unresolved',table_structure_error='invalid_cell_span')
    elif parser.rows:
        occupied, cells, rows = set(), [], []
        for row_number, row in enumerate(parser.rows,1):
            column = 1
            for cell in row:
                while (row_number,column) in occupied: column += 1
                value = ''.join(cell.pop('parts')).strip()
                cells.append(dict(row=row_number,column=column,text=value,**cell))
                for r in range(row_number,row_number+cell['rowspan']):
                    for c in range(column,column+cell['colspan']):
                        if (r,c) in occupied or r>len(parser.rows):
                            return dict(metadata,table_structure_status='unresolved',table_structure_error='overlapping_or_out_of_grid_span')
                        occupied.add((r,c))
                column += cell['colspan']
            width = max((c for r,c in occupied if r==row_number),default=0)
            rows.append([next((c['text'] for c in cells if c['row']==row_number and c['column']==col),'')
                         for col in range(1,width+1)])
        if strict_table and (not rows or len({len(row) for row in rows})!=1):
            return dict(metadata,table_structure_status='unresolved',table_structure_error='incomplete_grid')
        metadata.update(table_rows=rows,table_cells=cells,table_header_origin='surya_html_unreviewed',
                        table_structure_status='model_supplied')
    return metadata


def _visual_layout(crops, manager):
    from surya.layout import LayoutPredictor
    from PIL import Image
    frames, offsets = [], []
    for crop in crops:
        if crop.width<256 or crop.height<128:
            width,height=max(512,crop.width+32),max(512,crop.height+32)
            frame=Image.new('RGB',(width,height),'white')
            x,y=(width-crop.width)//2,(height-crop.height)//2
            frame.paste(crop,(x,y));frames.append(frame);offsets.append((x,y))
        else:
            frames.append(crop);offsets.append((0,0))
    detect=LayoutPredictor(manager)
    results=detect(frames)
    failed=[i for i,result in enumerate(results) if getattr(result,'error',False)]
    if failed:
        # Retry only failed image layouts, once. This does not descend into
        # nested pictures; translate the rescue margins back to source pixels.
        rescued=[]
        for i in failed:
            frame=Image.new('RGB',(frames[i].width+64,frames[i].height+64),'white')
            frame.paste(frames[i],(32,32));rescued.append(frame)
        print(json.dumps(dict(event='surya_visual_layout_retry',regions=failed,attempt=1)),flush=True)
        retries=detect(rescued)
        if len(retries)!=len(failed):raise RuntimeError('Visual layout retry count mismatch')
        for i,result in zip(failed,retries):
            results[i]=result
            offsets[i]=(offsets[i][0]+32,offsets[i][1]+32)
    for result,crop,(x,y) in zip(results,crops,offsets):
        if not (x or y):continue
        boxes=[]
        for box in result.bboxes:
            a,b,c,d=box.bbox
            a,b,c,d=max(0,a-x),max(0,b-y),min(crop.width,c-x),min(crop.height,d-y)
            if a>=c or b>=d:continue
            box.polygon=[[a,b],[c,b],[c,d],[a,d]]
            boxes.append(box)
        result.bboxes=boxes;result.image_bbox=[0,0,crop.width,crop.height]
    return results


def surya_visual_page(image, predictor, layout, *, reset_budget=True, high_resolution_image=None):
    """Run one extra layout -> block OCR pass inside each page image region.

    Each detected child is independent. Parent images never aggregate child
    text; IDs and bboxes retain the exact source containment and chunk boundary.
    """
    from .document_structure_service import normalize_layout
    manager = predictor.manager
    if reset_budget and isinstance(manager, SuryaStreamManager): manager.begin_page()
    started = time.perf_counter()
    calls_before = len(manager.calls) if isinstance(manager, SuryaStreamManager) else 0
    sharp = high_resolution_image or image
    if abs(sharp.width/sharp.height-image.width/image.height) > .01:
        raise ValueError('Visual source images must have the same page aspect ratio')
    sx, sy = sharp.width/image.width, sharp.height/image.height
    roots, children, pending = [], [], []

    def crop_for(box):
        from PIL import ImageDraw
        x0,y0=int(box[0]*sx),int(box[1]*sy)
        crop=sharp.crop((x0,y0,math.ceil(box[2]*sx),math.ceil(box[3]*sy)))
        draw=ImageDraw.Draw(crop)
        for text_box in text_boxes:
            left,top=max(box[0],text_box[0]),max(box[1],text_box[1])
            right,bottom=min(box[2],text_box[2]),min(box[3],text_box[3])
            if left<right and top<bottom:
                draw.rectangle((int(left*sx)-x0,int(top*sy)-y0,
                                math.ceil(right*sx)-x0-1,math.ceil(bottom*sy)-y0-1),fill='white')
        return crop

    def valid_box(box, width, height):
        return (len(box) == 4 and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box)
                and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)

    # A picture bbox can overlap page text. Preserve ownership of that text
    # instead of reading clipped fragments into the picture's child chunks.
    text_boxes=[r['bbox'] for r in layout
        if normalize_layout(r.get('label')) in {'title','paragraph','list_item','list_group',
            'figure_caption','table_caption','table','code','formula','page_header','page_footer'}
        and r.get('ocr_status','completed')=='completed'
        and valid_box(r.get('bbox',[]),image.width,image.height)]

    def belongs_to_page_text(box):
        area=(box[2]-box[0])*(box[3]-box[1])
        return area>0 and any(max(0,min(box[2],b[2])-max(box[0],b[0]))*
            max(0,min(box[3],b[3])-max(box[1],b[1]))/area>=.9 for b in text_boxes)

    for index, region in enumerate(layout):
        if normalize_layout(region.get('label')) != 'figure': continue
        box = region.get('bbox', [])
        if not valid_box(box, image.width, image.height):
            raise ValueError('Visual OCR requires a valid page-pixel bbox')
        root = dict(index=index, bbox=box, label=region['label'], status='completed', text='',
                    ocr_lines=[], error_type='', confidence=0, children=[], candidate_failures=0,
                    candidate_rejections=0, unreadable_lines=0, excluded_page_text=0, ignored_tiny_text=0)
        roots.append(root)
        pending.append((region.get('region_id', f'surya:{index}'), box, 1, root))
    if isinstance(manager, SuryaStreamManager): manager.visual_phase = True
    try:
        # Exactly one batch of page-level image regions. Inner images are never queued.
        for wave in [pending] if pending else []:
            crops = [crop_for(box) for _,box,_,_ in wave]
            print(json.dumps(dict(event='surya_visual_layout', depth=wave[0][2], regions=len(wave))), flush=True)
            try:
                layouts = _visual_layout(crops, manager)
                if len(layouts) != len(wave): raise RuntimeError('Visual layout result count mismatch')
            except Exception as error:
                for _,_,_,root in wave:
                    root['candidate_failures'] += 1; root['error_type'] = type(error).__name__
                continue
            # OCR only regions labelled as text/table/etc by this layout pass.
            # The same RecognitionPredictor skips inner images; they are saved as-is.
            usable = [(i, result) for i,result in enumerate(layouts) if not result.error]
            try:
                recognized = predictor([crops[i] for i,_ in usable],
                                      layout_results=[result for _,result in usable], full_page=False) if usable else []
                if len(recognized) != len(usable): raise RuntimeError('Visual OCR result count mismatch')
                by_index = {i:r for (i,_),r in zip(usable, recognized)}
            except Exception as error:
                by_index = {}
                for i,_ in usable:
                    wave[i][3]['candidate_failures'] += 1; wave[i][3]['error_type'] = type(error).__name__
            for i,(parent_id,parent_box,depth,root) in enumerate(wave):
                if layouts[i].error:
                    root['candidate_failures'] += 1; root['error_type'] = 'LayoutError'
                    continue
                result = by_index.get(i)
                if result is None: continue
                origin_x, origin_y = int(parent_box[0]*sx), int(parent_box[1]*sy)
                for child_index, (block, order) in enumerate(native_reading_order(result.blocks, scope=parent_id)):
                    # The extra pass extracts text only. Inner pictures are not
                    # source chunks, graph nodes, or inputs to Granite.
                    if normalize_layout(block.label) == 'figure':
                        continue
                    local = list(block.bbox)
                    if not valid_box(local, crops[i].width, crops[i].height):
                        root['candidate_failures'] += 1; continue
                    box = [(origin_x+local[0])/sx, (origin_y+local[1])/sy,
                           min(image.width,(origin_x+local[2])/sx), min(image.height,(origin_y+local[3])/sy)]
                    if belongs_to_page_text(box):
                        root['excluded_page_text'] += 1
                        continue
                    region_id = parent_id+'/'+str(child_index)
                    child = dict(region_id=region_id, parent_region_id=parent_id, bbox=box,
                                 label=block.label, confidence=float(getattr(block,'confidence',None) or 0),
                                 visual_depth=depth, chunk_policy='surya_layout_region',
                                 ocr_status='failed' if block.error else 'completed', **order)
                    children.append(child); root['children'].append(child)
                    if block.error:
                        root['candidate_failures'] += 1; continue
                    raw = block.html or ''
                    text = _html_text(raw)
                    if not text:
                        child['ocr_status']='empty'; continue
                    child['ocr_validation']='surya_single_pass_unverified'
                    child.update(source_html_metadata(raw))
                    root['ocr_lines'].append(dict(text=text,bbox=box,score=child['confidence'],region_id=region_id,
                        parent_region_id=parent_id,text_source='surya_image_layout_ocr',
                        text_bbox_scope='surya_layout_region'))
    finally:
        if isinstance(manager, SuryaStreamManager): manager.visual_phase = False
    for root in roots:
        incomplete = root['candidate_failures'] or root['candidate_rejections']
        root['status'] = ('partial' if root['ocr_lines'] else 'failed' if root['candidate_failures'] else 'unresolved') if incomplete else 'completed' if root['ocr_lines'] else 'no_text'
    metrics = dict(policy=VISUAL_OCR_POLICY, regions=len(roots), child_regions=len(children),
        excluded_page_text_regions=sum(r['excluded_page_text'] for r in roots),
        ignored_tiny_text_regions=sum(r['ignored_tiny_text'] for r in roots),
        failed=sum(bool(r['candidate_failures']) for r in roots),
        unreadable_regions=sum(r['status'] in {'partial','unresolved'} for r in roots),
        candidate_failures=sum(r['candidate_failures'] for r in roots),
        candidate_rejections=sum(r['candidate_rejections'] for r in roots),
        text_regions=sum(len(r['ocr_lines']) for r in roots), elapsed_s=round(time.perf_counter()-started,3))
    if isinstance(manager, SuryaStreamManager): metrics['generation_calls']=manager.calls[calls_before:]
    print(json.dumps(dict(event='surya_visual_ocr',status='partial' if metrics['failed'] or metrics['unreadable_regions'] else 'completed',
                         **{k:v for k,v in metrics.items() if k!='generation_calls'})),flush=True)
    return dict(regions=roots, metrics=metrics, cacheable=not metrics['failed'])


def merge_visual_ocr(page, supplemental):
    """Flatten the region tree in source order, retaining each child as a chunk."""
    roots = [dict(r,region_id=r.get('region_id',f'surya:{i}'),chunk_policy='surya_layout_region')
             for i,r in enumerate(page['layout'])]
    lines = []
    for line in page['ocr_lines']:
        item = dict(line)
        if not item.get('region_id'):
            matches = [r for r in roots if list(r['bbox']) == list(item['bbox'])]
            if len(matches) == 1: item['region_id'] = matches[0]['region_id']
        lines.append(item)
    # A large figure can overlap an already extracted page text region.
    # Deduplicate only equal text at the same location, never by text alone.
    def duplicate(candidate):
        text=' '.join(candidate.get('text','').casefold().split())
        a=candidate.get('bbox',[])
        if not text or len(a)!=4:return False
        for existing in lines:
            if text!=' '.join(existing.get('text','').casefold().split()):continue
            b=existing.get('bbox',[])
            if len(b)!=4:continue
            overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            largest=max((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
            if largest>0 and overlap/largest>=.8:return True
        return False
    descendants = {}
    duplicates=0
    for row in supplemental['regions']:
        region = roots[row['index']]
        if row['bbox'] != region['bbox'] or row['label'] != region['label']:
            raise ValueError('Visual OCR source region mismatch')
        region.update(visual_ocr_status=row['status'],visual_ocr_error=row['error_type'],
                      visual_ocr_candidate_failures=row.get('candidate_failures',0),
                      visual_ocr_candidate_rejections=row.get('candidate_rejections',0))
        duplicate_ids=set()
        for line in row.get('ocr_lines',[]):
            if duplicate(line):
                duplicate_ids.add(line['region_id']);duplicates+=1
            else:
                lines.append(line)
        for child in row.get('children',[]):
            if child['region_id'] in duplicate_ids:continue
            descendants.setdefault(child['parent_region_id'],[]).append(child)
    flattened = []
    def visit(region):
        region=dict(region,source_region_order=len(flattened))
        flattened.append(region)
        for child in descendants.get(region['region_id'],[]): visit(child)
    for region in roots: visit(region)
    result=dict(page,layout=flattened,ocr_lines=lines,
        metrics=dict(page.get('metrics',{}),visual_ocr=dict(supplemental['metrics'],duplicate_text_regions=duplicates),
                     elapsed_s=supplemental['metrics']['elapsed_s'],ocr_mode='single_image_layout_upgrade',
                     generation_calls=supplemental['metrics'].get('generation_calls',[])),
        visual_ocr_policy=VISUAL_OCR_POLICY,cacheable=supplemental['cacheable'])
    return result


def surya_page(image, predictor, high_resolution_image=None):
    """Normalize source OCR, including text inside previously skipped figures."""
    manager = getattr(predictor, 'manager', None)
    if isinstance(manager, SuryaStreamManager):
        manager.begin_page()
    started = time.perf_counter()
    mode = os.getenv('SURYA_OCR_MODE', 'full_page')
    if isinstance(manager, SuryaStreamManager) and mode == 'blocks':
        from surya.layout import LayoutPredictor
        layout = LayoutPredictor(manager)([image])
        if len(layout) != 1 or layout[0].error:
            raise RuntimeError('Surya could not detect source layout')
        if high_resolution_image is not None:
            from copy import deepcopy
            sx,sy=high_resolution_image.width/image.width,high_resolution_image.height/image.height
            if abs(sx/sy-1)>.01:raise ValueError('OCR images must have matching aspect ratios')
            sharp_layout=deepcopy(layout)
            for box in sharp_layout[0].bboxes:
                box.polygon=[[x*sx,y*sy] for x,y in box.polygon]
            sharp_layout[0].image_bbox=[0,0,high_resolution_image.width,high_resolution_image.height]
            result=predictor([high_resolution_image],layout_results=sharp_layout,full_page=False)[0]
            for block in result.blocks:
                block.polygon=[[x/sx,y/sy] for x,y in block.polygon]
        else:
            result = predictor([image], layout_results=layout, full_page=False)[0]
    elif mode in {'blocks', 'full_page'}:
        source_image = (high_resolution_image or image) if isinstance(manager,SuryaStreamManager) else image
        result = predictor([source_image], full_page=True)[0]
        if source_image is not image:
            sx,sy=source_image.width/image.width,source_image.height/image.height
            for block in result.blocks:
                block.polygon=[[x/sx,y/sy] for x,y in block.polygon]
    else:
        raise ValueError('SURYA_OCR_MODE must be blocks or full_page')
    regions, lines = ([], [])
    source = 'surya_layout_model' if isinstance(manager, SuryaStreamManager) and mode == 'blocks' else 'surya_full_page_model'
    for block, order in native_reading_order(result.blocks, source=source):
        bbox = [float(value) for value in block.bbox]
        score = float(getattr(block, 'confidence', None) or 0)
        region_id = f'surya:{len(regions)}'
        regions.append(dict(bbox=bbox,label=block.label,confidence=score,region_id=region_id,
                            chunk_policy='surya_layout_region',ocr_status='failed' if block.error else 'skipped' if block.skipped else 'completed',
                            source_region_order=len(regions), **order))
        if block.skipped or block.error:
            continue
        regions[-1].update(source_html_metadata(block.html))
        text = _html_text(block.html)
        if text:
            lines.append(dict(text=text,score=score,bbox=bbox,region_id=region_id))
    metrics = dict(elapsed_s=round(time.perf_counter()-started, 3),body_ocr_policy=BODY_OCR_POLICY,
                   failed_regions=sum(r['ocr_status']=='failed' for r in regions),
                   reading_order=dict(source=source, scope='page',
                       status='unresolved' if any(r['reading_order_unresolved'] for r in regions) else 'model_predicted',
                       regions=len(regions)))
    if isinstance(manager, SuryaStreamManager):
        metrics['generation_calls'] = list(manager.calls)
        metrics['ocr_mode'] = mode
    page = dict(layout=regions,ocr_lines=lines,model='surya-ocr-v2',metrics=metrics)
    from .document_structure_service import normalize_layout
    if any(normalize_layout(r['label'])=='figure' for r in regions):
        page = merge_visual_ocr(page,surya_visual_page(image,predictor,regions,reset_budget=False,
                                                   high_resolution_image=high_resolution_image))
        page['metrics']['elapsed_s'] = round(time.perf_counter()-started,3)
        page['metrics']['ocr_mode'] = mode
        if isinstance(manager,SuryaStreamManager): page['metrics']['generation_calls'] = list(manager.calls)
    else:
        page.update(visual_ocr_policy=VISUAL_OCR_POLICY,cacheable=True)
    page['cacheable'] = page.get('cacheable',True) and not metrics['failed_regions']
    return page
