import os
import xml.sax.saxutils as saxutils
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors


def _get_unicode_font():
    font_name = 'Helvetica'
    CANDIDATE_FONTS = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/times.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    ]
    for fpath in CANDIDATE_FONTS:
        if os.path.exists(fpath):
            try:
                pdfmetrics.registerFont(TTFont('UnicodeFont', fpath))
                return 'UnicodeFont'
            except Exception:
                pass
    return font_name


def generate_pdf_from_text(title: str, contract_code: str, text_content: str) -> bytes:
    font_name = _get_unicode_font()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=16,
        leading=22,
        textColor=colors.HexColor('#1e293b'),
        alignment=1,  # Center
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748b'),
        alignment=1,  # Center
        spaceAfter=12
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    story = []

    escaped_title = saxutils.escape(title or 'HỢP ĐỒNG')
    story.append(Paragraph(escaped_title, title_style))
    if contract_code:
        story.append(Paragraph(f'Mã hợp đồng: {saxutils.escape(contract_code)}', meta_style))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=4, spaceAfter=14))

    if text_content:
        lines = text_content.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 6))
                continue
            escaped_line = saxutils.escape(stripped)
            story.append(Paragraph(escaped_line, body_style))
    else:
        story.append(Paragraph("(Không có nội dung)", body_style))

    doc.build(story)
    return buffer.getvalue()
