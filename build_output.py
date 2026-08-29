"""
build_output.py
----------------
Writes the translated glossary to CSV and a formatted bilingual PDF.

Each entry is rendered as:
    N.  TERM : definition text, properly word-wrapped so nothing
        runs off the page edge.
    N.  translated-term : translated-definition text

The term is upper-cased and followed by " : " so it's unmistakably
separated from the definition -- no more guessing where one ends and
the other begins.

Uses a Unicode font (Noto Sans Tamil / Malayalam) so Indic scripts
render correctly -- the default PDF fonts do NOT support them.

Download fonts once (not included here due to license/size):
    Noto Sans Tamil:      https://fonts.google.com/noto/specimen/Noto+Sans+Tamil
    Noto Sans Malayalam:  https://fonts.google.com/noto/specimen/Noto+Sans+Malayalam
Place the .ttf file next to this script, or pass a custom path.
"""

import csv
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def write_csv(rows: list[dict], out_path: str):
    fieldnames = ["term_en", "definition_en", "term_translated", "definition_translated"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV -> {out_path}")


def _wrap_text(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    """Word-wrap text to fit max_width, using actual glyph widths so
    nothing gets silently cut off at the page edge."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            # A single word longer than max_width: hard-break it
            if pdfmetrics.stringWidth(word, font_name, font_size) > max_width:
                piece = ""
                for ch in word:
                    if pdfmetrics.stringWidth(piece + ch, font_name, font_size) <= max_width:
                        piece += ch
                    else:
                        lines.append(piece)
                        piece = ch
                current = piece
            else:
                current = word
    if current:
        lines.append(current)
    return lines or [""]


def write_pdf(rows: list[dict], out_path: str, font_path: str, font_name: str = "IndicFont"):
    pdfmetrics.registerFont(TTFont(font_name, font_path))

    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    usable_width = width - 2 * margin
    indent = 8 * mm  # hanging indent for wrapped continuation lines
    line_height = 14
    body_size = 10
    title_size = 16

    y = height - margin
    c.setFont(font_name, title_size)
    c.drawString(margin, y, "Bilingual Glossary")
    y -= line_height * 2.2

    def draw_entry(number: int, label: str, term: str, definition: str):
        nonlocal y
        c.setFont(font_name, body_size)

        prefix = f"{number}. [{label}] {term.upper()} : "
        prefix_width = pdfmetrics.stringWidth(prefix, font_name, body_size)

        # First line: number + label + term + colon + as much definition as fits
        first_line_budget = usable_width - prefix_width
        if first_line_budget < usable_width * 0.25:
            # Term itself is long -- put definition on its own wrapped block
            if y < margin + line_height:
                c.showPage()
                c.setFont(font_name, body_size)
                y = height - margin
            c.drawString(margin, y, prefix.rstrip())
            y -= line_height
            def_lines = _wrap_text(definition, font_name, body_size, usable_width - indent)
        else:
            wrapped = _wrap_text(definition, font_name, body_size, first_line_budget)
            if y < margin + line_height:
                c.showPage()
                c.setFont(font_name, body_size)
                y = height - margin
            c.drawString(margin, y, prefix + wrapped[0])
            y -= line_height
            def_lines = wrapped[1:]
            if def_lines:
                # Re-wrap the remainder at full indented width for consistency
                remainder = " ".join(def_lines)
                def_lines = _wrap_text(remainder, font_name, body_size, usable_width - indent)

        for line in def_lines:
            if y < margin + line_height:
                c.showPage()
                c.setFont(font_name, body_size)
                y = height - margin
            c.drawString(margin + indent, y, line)
            y -= line_height

    for i, row in enumerate(rows, start=1):
        draw_entry(i, "EN", row["term_en"], row["definition_en"])
        draw_entry(i, "TR", row["term_translated"], row["definition_translated"])
        y -= line_height * 0.6  # gap between entries
        if y < margin:
            c.showPage()
            c.setFont(font_name, body_size)
            y = height - margin

    c.save()
    print(f"Wrote PDF -> {out_path}")