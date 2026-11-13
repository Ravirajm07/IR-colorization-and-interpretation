from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.utils.config import get_settings


def generate_pdf_report(job: dict, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm)
    story = [
        Paragraph("IR WARRIORS", styles["Title"]),
        Paragraph("AI-Based Infrared Satellite Image Enhancement and RGB Colorization", styles["Heading2"]),
        Paragraph("Scientific description: IR enhancement and RGB translation", styles["Italic"]),
        Spacer(1, 8 * mm),
        Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()}", styles["BodyText"]),
        Paragraph(f"Job ID: {job['job_id']}", styles["BodyText"]),
        Spacer(1, 5 * mm),
        Paragraph("Scene Metadata", styles["Heading2"]),
        _dict_table(job.get("input_metadata", {})),
        Spacer(1, 5 * mm),
        Paragraph("Model and Metrics", styles["Heading2"]),
        Paragraph(f"Model: {get_settings().model_version}", styles["BodyText"]),
        _dict_table(job.get("metrics", {}) or {"Metrics": "Metric unavailable"}),
    ]
    for label, key in (("Visual Comparison", "comparison"), ("GIS Boundary Overlay", "boundary_overlay"), ("Generated Preview", "preview")):
        path = job.get("output_paths", {}).get(key)
        if path and Path(path).exists():
            story.extend([PageBreak(), Paragraph(label, styles["Heading2"]), Image(path, width=170 * mm, height=105 * mm, kind="proportional")])
    story.extend(
        [
            Spacer(1, 8 * mm),
            Paragraph("Limitations", styles["Heading2"]),
            Paragraph(
                "This prototype performs learned thermal-to-visible translation. Generated color is inferential, not directly observed. "
                "Landsat resolution is not suitable for reliable vehicle-level detection, and this system does not claim true super-resolution. "
                "Metrics describe only the evaluated held-out scenes and are not evidence of nationwide operational performance.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story)
    return output


def _dict_table(values: dict) -> Table:
    rows = [[str(key).replace("_", " ").title(), _display(value)] for key, value in values.items()]
    table = Table(rows or [["Status", "Metric unavailable"]], colWidths=[55 * mm, 110 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E7F3F8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0A2540")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#AFC8D5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#F8FBFC")]),
            ]
        )
    )
    return table


def _display(value: object) -> str:
    return json.dumps(value) if isinstance(value, (dict, list, tuple)) else str(value)


def create_model_bundle(output_path: str | Path) -> Path:
    settings = get_settings()
    candidates = ("best_generator.pth", "best_discriminator.pth", "config.json", "training_history.csv")
    available = [settings.checkpoint_dir / name for name in candidates if (settings.checkpoint_dir / name).exists()]
    if not available:
        raise FileNotFoundError("No trained model artifacts are available.")
    with ZipFile(output_path, "w", ZIP_DEFLATED) as archive:
        for path in available:
            archive.write(path, path.name)
    return Path(output_path)


def generate_evaluation_pdf(summary: dict, comparison_path: str | Path, output_path: str | Path) -> Path:
    job = {
        "job_id": "held-out-test-evaluation",
        "input_metadata": {"evaluation_split": "test", "metric_source": "real model predictions"},
        "metrics": summary,
        "output_paths": {"comparison": str(comparison_path)},
    }
    return generate_pdf_report(job, output_path)
