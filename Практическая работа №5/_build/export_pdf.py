"""Экспорт выполненного блокнота в PDF: nbconvert -> HTML -> печать Edge headless."""
import subprocess, tempfile, shutil
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "Практика_5.ipynb"
PDF = ROOT / "Практика_5_ДицельНС_ИКБО-11-23.pdf"
HTML = ROOT / "_build" / "Практика_5.html"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

PRINT_CSS = """
<style>
@page { size: A4; margin: 14mm 12mm 14mm 12mm; }
body { font-family: "Segoe UI", Arial, sans-serif; }
.jp-Notebook, body.jp-Notebook { padding: 0 !important; }
pre, .jp-OutputArea-output pre, .highlight pre, .jp-RenderedText pre {
  white-space: pre-wrap !important; word-break: break-word !important; overflow: visible !important;
  font-size: 8.6pt !important; line-height: 1.32 !important; }
.jp-InputPrompt, .jp-OutputPrompt { min-width: 0 !important; width: 0 !important; padding: 0 !important; visibility: hidden; }
.jp-Cell { padding: 2px 0 !important; }
.jp-RenderedImage img, .jp-OutputArea-output img { max-width: 100% !important; height: auto !important; break-inside: avoid; }
.jp-OutputArea-output { break-inside: auto; }
.jp-RenderedHTMLCommon table { font-size: 9pt; }
.jp-RenderedHTMLCommon h1 { font-size: 20pt; }
.jp-RenderedHTMLCommon h2 { break-before: page; font-size: 16pt; }
.jp-RenderedHTMLCommon h3 { font-size: 13pt; break-after: avoid; }
.jp-RenderedHTMLCommon { font-size: 10.5pt; }
</style>
"""

nb = nbformat.read(NB, as_version=4)
exporter = HTMLExporter(template_name="lab")
body, _ = exporter.from_notebook_node(nb)
body = body.replace("</head>", PRINT_CSS + "</head>", 1)
HTML.write_text(body, encoding="utf-8")

profile = Path(tempfile.mkdtemp(prefix="edge_pdf_"))
try:
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
                    "--no-pdf-header-footer", "--virtual-time-budget=20000",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, timeout=300)
finally:
    shutil.rmtree(profile, ignore_errors=True)
print("PDF:", PDF, PDF.stat().st_size // 1024, "KiB")
