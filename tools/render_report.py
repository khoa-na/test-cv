"""Render REPORT.md thành PDF A4 bằng Markdown + WeasyPrint."""

from __future__ import annotations

import argparse
from pathlib import Path

from markdown import markdown
from weasyprint import HTML


CSS = """
@page {
  size: A4;
  margin: 14mm 14mm 15mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    color: #64748b;
    font: 8pt "DejaVu Sans";
  }
}
html { font-family: "DejaVu Sans", sans-serif; color: #172033; }
body { font-size: 9pt; line-height: 1.32; }
h1 {
  font-size: 22pt; line-height: 1.15; color: #0f172a;
  border-bottom: 3px solid #2563eb; padding-bottom: 7mm; margin: 0 0 9mm;
}
h2 {
  font-size: 15pt; color: #0f3d70; border-bottom: 1px solid #cbd5e1;
  padding-bottom: 2mm; margin: 7mm 0 3mm;
}
h3 { font-size: 11.5pt; color: #1e3a5f; margin: 4.5mm 0 2mm; }
h4 { font-size: 10pt; color: #334155; margin: 4mm 0 1.5mm; }
p { margin: 0 0 2.4mm; orphans: 3; widows: 3; }
ul, ol { margin: 1.5mm 0 3mm 5mm; padding-left: 4mm; }
li { margin-bottom: 1mm; }
body > ol:last-child { font-size: 7.2pt; line-height: 1.17; margin-bottom: 0; }
body > ol:last-child li { margin-bottom: 0.35mm; }
strong { color: #0f172a; }
a { color: #155eaa; text-decoration: none; overflow-wrap: anywhere; }
code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 8pt;
  background: #f1f5f9; padding: 0.15em 0.3em; border-radius: 2px;
}
pre {
  font-family: "DejaVu Sans Mono", monospace; font-size: 7.4pt;
  line-height: 1.28; background: #f8fafc; border: 1px solid #dbe3ed;
  padding: 3mm; white-space: pre-wrap; overflow-wrap: anywhere;
  break-inside: avoid;
}
pre code { background: transparent; padding: 0; }
table {
  width: 100%; border-collapse: collapse; margin: 2.5mm 0 4mm;
  font-size: 7.6pt; line-height: 1.22;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th {
  background: #eaf2fb; color: #0f2f52; font-weight: 700;
  border: 1px solid #9fb3c8; padding: 1.5mm;
}
td { border: 1px solid #b8c6d5; padding: 1.35mm; vertical-align: top; }
tbody tr:nth-child(even) { background: #f8fafc; }
img {
  display: block; max-width: 100%; max-height: 142mm;
  object-fit: contain; margin: 3mm auto 4mm;
}
img[src$=".svg"] { max-height: 92mm; }
img[src*="MaskPR_curve"] { max-height: 103mm; }
blockquote {
  margin: 3mm 0; padding: 2mm 4mm; border-left: 3px solid #3b82f6;
  background: #eff6ff; color: #334155;
}
.keep-with-next { break-after: avoid; }
.keep-with-next + p { break-inside: avoid; }
"""


def render(source: Path, output: Path) -> None:
    body = markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    document = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Engineering Decision Report - Computer Vision for EV</title>
  <style>{CSS}</style>
</head>
<body>{body}</body>
</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=document, base_url=str(source.parent.resolve())).write_pdf(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("REPORT.md"))
    parser.add_argument(
        "--output", type=Path, default=Path("REPORT.pdf")
    )
    args = parser.parse_args()
    render(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
