#!/usr/bin/env python3
"""Generate A4 vector PDFs for AprilRobotics tagStandard41h12 tags.

Important for tagStandard41h12:
  - total_width = 9 modules (the whole bitmap in apriltag-imgs)
  - width_at_border = 5 modules (the pose-estimation tag_size edge)
So a requested tag_size_mm of 60 mm prints a 108 mm-wide 9x9 bitmap.
Use --tag-size 0.060 with action_go for that PDF.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image

MM_TO_PT = 72.0 / 25.4
A4_W_MM = 210.0
A4_H_MM = 297.0
A4_W_PT = A4_W_MM * MM_TO_PT
A4_H_PT = A4_H_MM * MM_TO_PT
TOTAL_WIDTH = 9
WIDTH_AT_BORDER = 5

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def esc_pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfPage:
    def __init__(self):
        self.ops: list[str] = []

    def rect_fill_mm(self, x, y, w, h, gray=0.0):
        # y is top-left in mm; PDF uses bottom-left.
        xp = x * MM_TO_PT
        yp = A4_H_PT - (y + h) * MM_TO_PT
        wp = w * MM_TO_PT
        hp = h * MM_TO_PT
        self.ops.append(f"{gray:.3f} g {xp:.3f} {yp:.3f} {wp:.3f} {hp:.3f} re f")

    def rect_stroke_mm(self, x, y, w, h, gray=0.65, lw=0.25, dash=None):
        xp = x * MM_TO_PT
        yp = A4_H_PT - (y + h) * MM_TO_PT
        wp = w * MM_TO_PT
        hp = h * MM_TO_PT
        dash_op = "[] 0 d" if dash is None else f"[{dash[0]} {dash[1]}] 0 d"
        self.ops.append(f"{gray:.3f} G {lw:.3f} w {dash_op} {xp:.3f} {yp:.3f} {wp:.3f} {hp:.3f} re S [] 0 d")

    def line_mm(self, x1, y1, x2, y2, gray=0.0, lw=0.5):
        x1p = x1 * MM_TO_PT
        y1p = A4_H_PT - y1 * MM_TO_PT
        x2p = x2 * MM_TO_PT
        y2p = A4_H_PT - y2 * MM_TO_PT
        self.ops.append(f"{gray:.3f} G {lw:.3f} w {x1p:.3f} {y1p:.3f} m {x2p:.3f} {y2p:.3f} l S")

    def text_mm(self, x, y, text, size=10, gray=0.0):
        # y is top baseline-ish in mm.
        xp = x * MM_TO_PT
        yp = A4_H_PT - y * MM_TO_PT
        self.ops.append(f"BT /F1 {size:.1f} Tf {gray:.3f} g {xp:.3f} {yp:.3f} Td ({esc_pdf_text(text)}) Tj ET")

    def content(self) -> bytes:
        return ("\n".join(self.ops) + "\n").encode("ascii")


def write_pdf(path: Path, page: PdfPage):
    stream = page.content()
    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    page_obj = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {A4_W_PT:.3f} {A4_H_PT:.3f}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode("ascii")
    objs.append(page_obj)
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream")

    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(data))
        data.extend(f"{i} 0 obj\n".encode("ascii"))
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objs)+1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        data.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    data.extend(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    path.write_bytes(bytes(data))


def load_matrix(tag_id: int):
    img = Image.open(SRC / f"tag41_12_{tag_id:05d}.png").convert("L")
    if img.size != (TOTAL_WIDTH, TOTAL_WIDTH):
        raise ValueError(f"unexpected tag image size {img.size}")
    return [[img.getpixel((x, y)) < 128 for x in range(TOTAL_WIDTH)] for y in range(TOTAL_WIDTH)]


def draw_tag(page: PdfPage, tag_id: int, tag_size_mm: float, cx_mm: float, y_top_mm: float, label: bool = True):
    mat = load_matrix(tag_id)
    module_mm = tag_size_mm / WIDTH_AT_BORDER
    code_mm = TOTAL_WIDTH * module_mm
    x0 = cx_mm - code_mm / 2.0

    # Draw only black modules; the A4 paper is the white modules and quiet zone.
    for y in range(TOTAL_WIDTH):
        for x in range(TOTAL_WIDTH):
            if mat[y][x]:
                page.rect_fill_mm(x0 + x * module_mm, y_top_mm + y * module_mm, module_mm, module_mm, gray=0.0)

    quiet_mm = module_mm
    page.rect_stroke_mm(x0 - quiet_mm, y_top_mm - quiet_mm, code_mm + 2 * quiet_mm, code_mm + 2 * quiet_mm,
                        gray=0.72, lw=0.35, dash=(3, 3))

    if label:
        ty = y_top_mm + code_mm + quiet_mm + 8
        page.text_mm(x0 - quiet_mm, ty, f"tagStandard41h12  id={tag_id}  tag_size={tag_size_mm:.0f} mm", size=10)
        page.text_mm(x0 - quiet_mm, ty + 5, f"whole 9x9 code width = {code_mm:.1f} mm; run: --tag-id {tag_id} --tag-size {tag_size_mm/1000:.3f}", size=8, gray=0.15)
        page.text_mm(x0 - quiet_mm, ty + 10, "Print at 100% / Actual size. Do NOT fit-to-page.", size=8, gray=0.15)
    return code_mm, module_mm


def draw_scale_bars(page: PdfPage, x: float, y: float, tag_size_mm: float):
    # Calibration scale: user can verify the printer did not rescale.
    page.text_mm(x, y, "Printer scale check:", size=9)
    page.line_mm(x, y + 5, x + tag_size_mm, y + 5, gray=0, lw=0.7)
    page.line_mm(x, y + 3.5, x, y + 6.5, gray=0, lw=0.7)
    page.line_mm(x + tag_size_mm, y + 3.5, x + tag_size_mm, y + 6.5, gray=0, lw=0.7)
    page.text_mm(x, y + 10, f"this line should measure {tag_size_mm:.0f} mm", size=8)
    page.line_mm(x, y + 17, x + 100, y + 17, gray=0, lw=0.7)
    for t in range(0, 101, 10):
        tick_h = 3 if t % 50 else 5
        page.line_mm(x + t, y + 17 - tick_h / 2, x + t, y + 17 + tick_h / 2, gray=0, lw=0.45)
    page.text_mm(x, y + 23, "100 mm ruler", size=8)


def make_single(tag_size_mm: float):
    page = PdfPage()
    page.text_mm(20, 18, "AprilRobotics AprilTag printable sheet", size=15)
    page.text_mm(20, 26, "Family: tagStandard41h12   ID: 0", size=11)
    page.text_mm(20, 33, "Use the action_go value printed below; for tagStandard41h12 tag_size is 5 modules, while the visible code is 9 modules.", size=8)
    module_mm = tag_size_mm / WIDTH_AT_BORDER
    code_mm = TOTAL_WIDTH * module_mm
    y_top = 56 if tag_size_mm <= 70 else 52
    draw_tag(page, 0, tag_size_mm, A4_W_MM / 2, y_top, label=True)
    draw_scale_bars(page, 20, 254, tag_size_mm)
    out = ROOT / f"tagStandard41h12_id0_tagsize{int(tag_size_mm)}mm_A4.pdf"
    write_pdf(out, page)
    return out


def make_multi(tag_size_mm: float = 40.0):
    page = PdfPage()
    page.text_mm(16, 16, "AprilRobotics tagStandard41h12 multi-ID sheet", size=14)
    page.text_mm(16, 24, "For first action_go deployment use ONE known tag/offset. Multi-tag fusion is a later robustness upgrade.", size=8)
    positions = [(56, 50), (154, 50), (56, 167), (154, 167)]
    for tag_id, (cx, y_top) in enumerate(positions):
        draw_tag(page, tag_id, tag_size_mm, cx, y_top, label=True)
    draw_scale_bars(page, 16, 265, tag_size_mm)
    out = ROOT / f"tagStandard41h12_ids0-3_tagsize{int(tag_size_mm)}mm_A4.pdf"
    write_pdf(out, page)
    return out


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    outs = [make_single(s) for s in (50.0, 60.0, 70.0, 80.0)]
    outs.append(make_multi(40.0))
    print("Generated:")
    for p in outs:
        print(" -", p)


if __name__ == "__main__":
    main()
