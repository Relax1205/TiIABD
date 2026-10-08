import sys
import nbformat as nbf

src, dst = sys.argv[1], sys.argv[2]
cells, kind, buf = [], None, []

def flush():
    if kind is None:
        return
    text = "\n".join(buf).strip("\n")
    cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))

for line in open(src, encoding="utf-8").read().splitlines():
    if line.startswith("#%% "):
        flush()
        kind, buf = line[4:].strip(), []
    else:
        buf.append(line)
flush()

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, dst)
