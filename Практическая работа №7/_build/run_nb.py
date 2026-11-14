"""Выполнение блокнота в локальном окружении .venv с сохранением прогресса."""
from pathlib import Path
import os
import time
import nbformat
from nbclient import NotebookClient
from datetime import datetime
ROOT=Path(__file__).resolve().parent.parent
VENV=ROOT.parent/'.venv'
os.environ['JUPYTER_PATH']=str(VENV/'share/jupyter')
path=ROOT/'Практика_7.ipynb'
nb=nbformat.read(path,as_version=4)
def saved(cell,cell_index,**kwargs):
    nbformat.write(nb,path)
    print(datetime.now().isoformat(timespec='seconds'),'DONE',cell_index,flush=True)
def started(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(datetime.now().isoformat(timespec='seconds'),'START',cell_index,cell.source.splitlines()[0][:80],flush=True)
    if cell.cell_type=='code' and cell.source.startswith('variant='):
        # Предварительная загрузка архива идёт одновременно с основной частью.
        # Сам блокнот сохраняет стандартный download=True для обычного запуска.
        deadline=time.monotonic()+600
        archive=ROOT/'practice7_data/cifar-10-python.tar.gz'
        while not archive.exists():
            if time.monotonic()>deadline:raise TimeoutError('Не завершена предварительная загрузка CIFAR-10')
            time.sleep(1)
client=NotebookClient(nb,kernel_name='practice7',timeout=1800,
    resources={'metadata':{'path':str(ROOT)}},
    on_cell_start=started,on_cell_executed=saved)
try:client.execute()
finally:nbformat.write(nb,path)
