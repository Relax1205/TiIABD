"""Предварительное скачивание двух архивов; блокнот умеет загружать их сам."""
from pathlib import Path
import hashlib
import subprocess
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'practice7_data'
DATA.mkdir(exist_ok=True)
FILES=[
    ('bloodmnist.npz','https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1','7053d0359d879ad8a5505303e11de1dc'),
    ('MNIST/raw/train-images-idx3-ubyte.gz','https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz','f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('MNIST/raw/train-labels-idx1-ubyte.gz','https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz','d53e105ee54ea40749a09fcbcd1e9432'),
    ('MNIST/raw/t10k-images-idx3-ubyte.gz','https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz','9fb629c4189551a2d022fa330f9573f3'),
    ('MNIST/raw/t10k-labels-idx1-ubyte.gz','https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz','ec29112dd5afa0611ce80d1b7f02629c')
]
def get(item):
    name,url,expected=item
    dest=DATA/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest()==expected:
        print('Cached',name,flush=True);return
    temp=dest.with_name(dest.name+'.part')
    subprocess.run(['curl.exe','--fail','--location','--silent','--show-error',
        '--max-time','180','--retry','2','--output',str(temp),url],check=True,timeout=600)
    digest=hashlib.md5(temp.read_bytes()).hexdigest()
    assert digest==expected,(name,digest)
    temp.replace(dest)
    if name.endswith('.gz') and not name.endswith('.tar.gz'):
        with gzip.open(dest,'rb') as src,dest.with_suffix('').open('wb') as target:
            shutil.copyfileobj(src,target)
    print('Downloaded',name,dest.stat().st_size,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(get,FILES))
import sys
subprocess.run([sys.executable,str(ROOT/'_build/download_cifar.py')],check=True)
