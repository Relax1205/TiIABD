"""Выполнение с сохранением каждой завершённой ячейки и журналом прогресса."""
from pathlib import Path
from datetime import datetime
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parent.parent
path = ROOT / 'Практика_6.ipynb'
nb = nbformat.read(path, as_version=4)
def saved(cell, cell_index, **kwargs):
    nbformat.write(nb, path)
    print(datetime.now().isoformat(timespec='seconds'), 'DONE', cell_index, cell.source.splitlines()[0][:90], flush=True)
def start(cell, cell_index, **kwargs):
    print(datetime.now().isoformat(timespec='seconds'), 'START', cell_index, flush=True)
client = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}},
                        on_cell_executed=saved, on_cell_start=start)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
