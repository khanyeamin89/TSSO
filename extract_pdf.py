"""
One-time helper: extract text from a PDF into document.txt, with page
markers ([PAGE N]) so the Q&A app can cite page numbers.

Usage:
    python extract_pdf.py path/to/your.pdf
    python extract_pdf.py path/to/your.pdf --out document.txt
"""

import argparse
from pypdf import PdfReader


def extract(pdf_path: str, out_path: str) -> None:
    reader = PdfReader(pdf_path)
    parts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        parts.append(f"[PAGE {i + 1}]\n{text}")
    full_text = "\n".join(parts)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Extracted {len(reader.pages)} pages ({len(full_text):,} characters) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF text with page markers.")
    parser.add_argument("pdf_path", help="Path to the source PDF")
    parser.add_argument("--out", default="document.txt", help="Output text file path")
    args = parser.parse_args()
    extract(args.pdf_path, args.out)
