"""Скачивание архива CIFAR-10 с проверкой авторской контрольной суммы."""
from pathlib import Path
import hashlib,subprocess
ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'practice7_data'
DEST=DATA/'cifar-10-python.tar.gz'
SIZE=170498071
EXPECTED='c58f30108f718f92721af3b95e74349a'
URLS=[
    'https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz',
    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
]
if DEST.exists() and hashlib.md5(DEST.read_bytes()).hexdigest()==EXPECTED:
    print('CIFAR already verified',flush=True)
else:
    target=DATA/'cifar_complete.tmp'
    for url in URLS:
        result=subprocess.run(['curl.exe','--fail','--location','--silent','--show-error',
            '--max-time','180','--retry','1','--output',str(target),url],timeout=400)
        if result.returncode==0 and target.stat().st_size==SIZE and hashlib.md5(target.read_bytes()).hexdigest()==EXPECTED:
            target.replace(DEST)
            print('CIFAR ready and author MD5 verified',flush=True)
            break
    else:raise RuntimeError('Не удалось получить архив CIFAR-10 с верной контрольной суммой')
