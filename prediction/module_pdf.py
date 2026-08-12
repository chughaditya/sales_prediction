"""
module_pdf.py
---------------
Generates a clean, single-page-ish PDF report for a specialized module
prediction (Company Sales / House Price / Retail Sales), using reportlab
so it works without any extra system dependencies.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

BRAND_PURPLE = colors.HexColor("#7C3AED")
MUTED = colors.HexColor("#6B7280")


def generate_module_pdf(title: str, headline_label: str, headline_value: str,
                         inputs: dict, confidence: float, insights: list = None,
                         extra_metrics: dict = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                             leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=BRAND_PURPLE, spaceAfter=4)
    sub_style = ParagraphStyle("SubStyle", parent=styles["Normal"], textColor=MUTED, fontSize=9, spaceAfter=14)
    headline_style = ParagraphStyle("HeadlineStyle", parent=styles["Heading1"], fontSize=22, spaceAfter=2)
    label_style = ParagraphStyle("LabelStyle", parent=styles["Normal"], textColor=MUTED, fontSize=10, spaceAfter=14)
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
    body_style = styles["Normal"]

    story = [
        Paragraph(title, title_style),
        Paragraph(f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}", sub_style),
        Paragraph(headline_value, headline_style),
        Paragraph(headline_label, label_style),
    ]

    metrics = dict(extra_metrics or {})
    if confidence is not None:
        metrics["Model Confidence"] = f"{confidence:.1f}%"
    if metrics:
        story.append(Paragraph("Key Metrics", h2_style))
        rows = [[k, str(v)] for k, v in metrics.items()]
        table = Table(rows, colWidths=[7 * cm, 8 * cm])
        table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ]))
        story.append(table)

    if insights:
        story.append(Paragraph("AI Insights", h2_style))
        for point in insights:
            story.append(Paragraph(f"&bull; {point}", body_style))
            story.append(Spacer(1, 3))

    story.append(Paragraph("Input Parameters", h2_style))
    input_rows = [[str(k), str(v)] for k, v in inputs.items()]
    input_table = Table(input_rows, colWidths=[7 * cm, 8 * cm])
    input_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#F3F4F6")),
    ]))
    story.append(input_table)

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "AI Sales Prediction Pro &mdash; automatically generated report. Predictions are model "
        "estimates and should be used as directional guidance, not guaranteed outcomes.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7.5, textColor=MUTED),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
