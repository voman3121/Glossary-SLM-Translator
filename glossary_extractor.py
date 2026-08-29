"""
glossary_extractor.py
----------------------
Extracts "term : definition" pairs from an English glossary PDF.

Real-world PDFs wrap long definitions across multiple physical lines,
e.g.:

    Block: a large piece of a solid material that is square or
    rectangular in shape and usually has flat sides

If every line were parsed independently, that second line ("rectangular
in shape and usually has flat sides") would look like a new entry.
To avoid that, this extractor works in two passes:

    1. Detect the document's style ONCE by checking whether any line
       uses an explicit separator (":", "-", tab). If yes -> "separator"
       mode. If NO line anywhere uses a separator (some glossaries glue
       a single-word term directly onto its definition with nothing
       between them) -> "no_separator" mode, where a capitalized first
       word is treated as the term.
    2. Walk the lines once more. A line that matches the detected
       entry-start pattern begins a NEW entry. Any other line is
       assumed to be a WRAPPED CONTINUATION of the previous entry's
       definition and is appended to it, not treated as a new entry.

Lines that are just table/column headers (e.g. "Word Definition",
"Term", "Meaning") are skipped entirely.

Usage:
    python glossary_extractor.py input.pdf output.json
"""

import sys
import json
import re
import pdfplumber


# Explicit-separator entry start: "Term: definition", "Term - definition",
# or "Term<TAB>definition".
SEPARATOR_PATTERN = re.compile(
    r"^\s*(?P<term>[A-Za-z0-9][\w\s\-/&().]{1,60}?)\s*(?:[:\-–—]\s+|\t+)(?P<def>.{3,})$"
)

# No-separator entry start: single capitalized word immediately followed
# by its definition, e.g. "Block a large piece of a solid material...".
# Requires a capital first letter so wrapped continuation lines that
# happen to start with a lowercase word ("and usually has flat sides")
# are never mistaken for a new entry.
NO_SEPARATOR_PATTERN = re.compile(r"^\s*(?P<term>[A-Z][A-Za-z'\-]*)\s+(?P<def>.{8,})$")

# Lines that are table/column headers or document titles, not real entries.
HEADER_SKIP_LINES = {
    "word", "definition", "word definition", "term", "meaning",
    "term definition", "term meaning", "glossary", "bilingual glossary",
    "vocabulary", "word list", "en", "tr",
}


def _normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def extract_text_lines(pdf_path: str) -> list[str]:
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.split("\n"):
                stripped = raw.strip()
                if stripped:
                    lines.append(stripped)
    return lines


def _detect_mode(lines: list[str]) -> str:
    """Scan the whole document once to decide whether it uses explicit
    separators or not. A single confirmed separator match anywhere is
    enough evidence the whole doc follows that style."""
    for line in lines:
        if _normalize(line) in HEADER_SKIP_LINES:
            continue
        if SEPARATOR_PATTERN.match(line):
            return "separator"
    return "no_separator"


def parse_glossary(lines: list[str]) -> list[dict]:
    mode = _detect_mode(lines)
    pattern = SEPARATOR_PATTERN if mode == "separator" else NO_SEPARATOR_PATTERN

    entries = []
    for line in lines:
        if _normalize(line) in HEADER_SKIP_LINES:
            continue

        m = pattern.match(line)
        if m:
            entries.append({
                "term": m.group("term").strip(),
                "definition": m.group("def").strip(),
            })
        elif entries:
            # Wrapped continuation of the previous entry's definition
            entries[-1]["definition"] = (entries[-1]["definition"] + " " + line).strip()
        # else: stray line before any entry was found (e.g. a title
        # line the header skip-list didn't anticipate) -- drop it.

    return entries


def deduplicate(entries: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for e in entries:
        key = e["term"].lower()
        if key not in seen and len(e["term"]) > 1:
            seen.add(key)
            unique.append(e)
    return unique


def extract_glossary(pdf_path: str) -> list[dict]:
    lines = extract_text_lines(pdf_path)
    entries = parse_glossary(lines)
    return deduplicate(entries)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python glossary_extractor.py input.pdf output.json")
        sys.exit(1)

    pdf_path, out_path = sys.argv[1], sys.argv[2]
    glossary = extract_glossary(pdf_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(glossary)} glossary entries -> {out_path}")
    if glossary:
        print("Sample entry:", glossary[0])
    else:
        print("WARNING: no entries matched. Inspect the PDF layout and "
              "adjust the patterns in this file.")