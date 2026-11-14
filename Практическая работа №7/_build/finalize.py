"""Краткий отчёт по измеренным результатам и выводы в выполненном блокноте."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import nbformat
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'practice7_results'
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def table(frame):
    def fmt(value):
        if isinstance(value,float):return f'{value:.4f}'
        return str(value).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in frame.itertuples(index=False,name=None))
def image(name,caption):return f'![{caption}](practice7_results/{name})'
def worst_pair(name,names):
    pred=pd.read_csv(OUT/f'{name}_predictions.csv')
    cm=confusion_matrix(pred.label,pred.prediction,labels=range(len(names)))
    np.fill_diagonal(cm,0)
    i,j=np.unravel_index(cm.argmax(),cm.shape)
    return f'{names[i]} → {names[j]} ({cm[i,j]} объектов)'
env=load('environment.json')
mnist=load('mnist_metrics.json')
config=load('cifar_config.json')
cifar=load('cifar_metrics.json')
comparison=pd.read_csv(OUT/'bloodmnist_comparison.csv')
mh=pd.read_csv(OUT/'mnist_history.csv')
ch=pd.read_csv(OUT/'cifar_history.csv')
linear=comparison.loc[comparison['Модель']=='Линейная'].iloc[0]
cnn=comparison.loc[comparison['Модель']=='CNN'].iloc[0]
change=ch.valid_accuracy.diff()
end_epoch=int(ch.loc[change.idxmax(),'epoch'])
growth=float(change.max())
blood_names=['Базофил','Эозинофил','Эритробласт','Незрелые гранулоциты','Лимфоцит','Моноцит','Нейтрофил','Тромбоцит']
parts=[]
def add(s):parts.append(s.strip())
add('# Практическая работа №7. От тензоров PyTorch к классификации изображений\n\n**Дицель Н.С., ИКБО-11-23.** 07.11.2026. Проверил: ст. преподаватель Приходько Н.А.')
add(f'**Цель:** проверить вычисления PyTorch, обучить классификаторы MNIST, BloodMNIST и CIFAR-10. **Окружение:** Python {env["python"]}, PyTorch {env["torch"]}, torchvision {env["torchvision"]}, {env["device"]}, {env["cpu_threads"]} потока CPU. Seed=42, Adam(lr=0,001), batch=128, пять эпох. Test не используется для выбора архитектуры или эпохи.')
add('''## 1. Тензоры и шаг обучения

- Проверены арифметика, индексация, broadcasting, reshape/permute и общая память from_numpy.
- Ручное XWᵀ+b совпало с Linear; Linear(3,2) содержит 3×2+2=8 параметров.
- Два слоя без активации сведены в один. ReLU позволила аппроксимировать x².
- Ручные градиенты MSE и шаг SGD совпали с PyTorch; потеря уменьшилась.
- CrossEntropyLoss совпала с ручным расчётом; на вход подаются логиты.
- Проверены ручная свёртка и BatchNorm в train/eval. Pooling обучаемых весов не имеет.

**Вывод:** формы и режим вычисления важны отдельно от значений. zero_grad очищает градиенты, backward вычисляет их, step обновляет веса. Dataset возвращает один объект; DataLoader формирует пакеты.''')
add('## 2. MNIST: линейная модель')
add(f'''Использованы 10 000 train и 2 000 valid из исходного train; проверено отсутствие общих индексов. Официальный test — 10 000 изображений. Модель Flatten → Linear(784,10), **{mnist["parameters"]} параметров**.

После пятой эпохи: **accuracy test {mnist["accuracy"]:.2%}**, **macro-F1 {mnist["macro_f1"]:.4f}**. Потеря train изменилась с {mh.train_loss.iloc[0]:.4f} до {mh.train_loss.iloc[-1]:.4f}; accuracy valid — с {mh.valid_accuracy.iloc[0]:.2%} до {mh.valid_accuracy.iloc[-1]:.2%}.

**Вывод:** линейное правило по пикселям уже различает большинство цифр, но не строит локальные признаки. Веса выхода «3» визуализированы в блокноте; это вклад пикселя в логит, а не вероятность.''')
add(image('03_mnist_history.png','MNIST: обучение и valid'))
add('## 3. BloodMNIST: линейная модель и CNN')
add('''Общий train: по 600 изображений восьми классов, всего 4 800. Официальные valid — 1 712 и test — 3 421 сохранены целиком. Обе модели используют RGB 28×28 и одинаковую нормировку, порядок train-пакетов и число эпох.

CNN: Conv(3→8) → BN → ReLU → Pool → Conv(8→16) → BN → ReLU → Pool → Flatten → Linear(784→8). Проверено совпадение ответа через вектор признаков с полным проходом.''')
add(table(comparison))
add(f'''**Вывод:** CNN увеличила accuracy на {(cnn["Accuracy test"]-linear["Accuracy test"])*100:.2f} п.п. при меньшем числе параметров ({int(cnn["Параметров"])} против {int(linear["Параметров"])}). На этом CPU пакет 64 CNN обрабатывает в {cnn["Пакет 64, мс"]/linear["Пакет 64, мс"]:.1f} раза дольше: свёртки выполняют много операций над картами.

Время — медиана трёх повторов после прогрева; чтение и перенос данных исключены. Чаще всего CNN ошибалась в направлении **{worst_pair("blood_cnn",blood_names)}**. Число параметров, качество и скорость нужно оценивать отдельно.''')
add(image('09_blood_history.png','BloodMNIST: loss и valid'))
add(image('11_blood_confusion.png','BloodMNIST: матрица ошибок test'))
add(f'## 4. Самостоятельная часть: CIFAR-10, вариант №{config["variant"]}')
add(f'''Выбраны **{", ".join(config["class_names"]).lower()}**; исходные коды {config["original_classes"]} заменены на 0,1,2 в том же порядке. Правило назначения по группе/зачётке в материалах отсутствует; использован вариант №1 из заготовки.

Состав: **4 500 train / 900 valid / 900 test**. На класс — 1 500/300/300; train и valid взяты из исходного train без пересечения, test — из официального test. Индексы сохранены. Ошибочные упоминания 100 test в заготовке заменены требованием методички — 300.

Предобработка: Resize 32→28, ToTensor, Normalize(mean=0,5; std=0,5) по каналам, однократно. Пакет — (N,3,28,28), метки int64. CNN: два блока 3→8→16, pooling до 7×7, Flatten 784, Linear на три логита; **{cifar["parameters"]} параметр**.''')
add(table(ch[['epoch','train_loss','valid_accuracy','train_seconds']]))
add(f'''Параметры пятой эпохи: **accuracy test {cifar["accuracy"]:.2%}**, **macro-F1 {cifar["macro_f1"]:.4f}**. Постоянный ответ на сбалансированном test даёт 33,33%. Самый большой прирост valid между эпохами — **{end_epoch-1}→{end_epoch}: {growth*100:.2f} п.п.**''')
add(image('14_cifar_history.png','CIFAR-10: loss и valid'))
add(image('15_cifar_confusion.png','CIFAR-10: матрица ошибок test'))
add(f'''**Ошибки:** наиболее частое направление — **{worst_pair("cifar",config["class_names"])}**. Ниже показаны первые три неверных ответа, без отбора по красивому примеру. Подпись содержит исходный класс и предсказание; для показа нормировка обращена image×0,5+0,5.

Малая модель обучалась только пять эпох на части корпуса; результат относится к этому разбиению. График valid описывает процесс обучения, итоговый test не использовался для подбора.''')
add(image('16_cifar_errors.png','CIFAR-10: первые три ошибки'))
add('''**Сохранено:** выполненный блокнот, четыре state_dict, истории, метрики по классам, прогнозы, индексы разбиений, параметры опыта и рисунки. Для CIFAR-10 после загрузки весов логиты на 16 фиксированных test-примерах совпали. В проходах оценки веса и буферы BatchNorm не менялись.

**Ответы на вопросы:** отдельный файл «Ответы_на_вопросы_7.md» — 15 полных вопросов лекции №7 (слайды 70–71) и 7 вопросов раздела 8 практики, с краткими ответами.

Источники: выданные методичка и лекция №7; [PyTorch](https://pytorch.org/get-started/locally/), [MedMNIST](https://medmnist.com/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).''')
report='\n\n'.join(parts)+'\n'
(ROOT/'Результаты_практики_7.md').write_text(report,encoding='utf-8')
path=ROOT/'Практика_7.ipynb'
nb=nbformat.read(path,as_version=4)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for c in nb.cells for o in c.get('outputs',[]))
plain=re.sub(r'!\[[^\]]*\]\([^)]+\)','',report)
cell=next(c for c in nb.cells if c.cell_type=='markdown' and c.source.startswith('## 8. Итоговые результаты'))
cell.source='## 8. Итоговые результаты\n\n'+re.sub(r'(?m)^## ','### ',plain.split('\n',1)[1]).strip()
nbformat.validate(nb);nbformat.write(nb,path)
print('Отчёт подготовлен; все ячейки выполнены без ошибок.')
