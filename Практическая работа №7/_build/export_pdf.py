"""Краткий PDF-отчёт; полный код остаётся в Практика_7.ipynb."""
from pathlib import Path
import re,base64,tempfile,subprocess,shutil
import nbformat as nbf
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parent.parent
text=(ROOT/'Результаты_практики_7.md').read_text(encoding='utf-8')
cell=nbf.v4.new_markdown_cell(text)
cell.attachments={}
for name,relative in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)',text):
    path=ROOT/relative
    cell.attachments[path.name]={'image/png':base64.b64encode(path.read_bytes()).decode('ascii')}
    cell.source=cell.source.replace(f']({relative})',f'](attachment:{path.name})')
nb=nbf.v4.new_notebook(cells=[cell])
nbf.write(nb,ROOT/'_build/Краткий_отчёт_7.ipynb')
html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
css='''<style>
@page { size:A4; margin:13mm 14mm; }
body,.jp-RenderedHTMLCommon {font-family:"Segoe UI",Arial,sans-serif;font-size:10pt;line-height:1.4;}
.jp-Notebook {padding:0!important;} .jp-InputPrompt,.jp-OutputPrompt{display:none!important;}
h1{font-size:18pt!important;} h2{font-size:14pt!important;break-before:page;break-after:avoid;}
h2:first-of-type{break-before:auto;}
h3{break-after:avoid;} table{font-size:8.5pt!important;max-width:100%;}
img{max-width:100%!important;max-height:115mm;object-fit:contain;break-inside:avoid;}
p:has(img){text-align:center;break-inside:avoid;}
</style>'''
html=html.replace('</head>',css+'</head>',1)
file=ROOT/'_build/Практика_7.html';file.write_text(html,encoding='utf-8')
pdf=ROOT/'Практика_7_ДицельНС_ИКБО-11-23.pdf'
edge=r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
profile=Path(tempfile.mkdtemp(prefix='practice7_pdf_'))
try:
    subprocess.run([edge,'--headless=new','--disable-gpu',f'--user-data-dir={profile}',
        '--no-pdf-header-footer','--virtual-time-budget=15000',f'--print-to-pdf={pdf}',file.as_uri()],
        check=True,timeout=180)
finally:shutil.rmtree(profile,ignore_errors=True)
assert pdf.exists()
print('PDF:',pdf.stat().st_size//1024,'KiB')
