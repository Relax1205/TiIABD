"""Сборка полного решения ПР7; исходные выданные блокноты не изменяются."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent.parent
source=nbf.read(ROOT/'Практика_7_PyTorch_от_тензоров_к_CNN.ipynb',as_version=4)
C=[]
def md(s):C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):C.append(nbf.v4.new_code_cell(s.strip()))
def original(*indices):
    return '\n\n'.join(source.cells[i].source for i in indices)

md("""# Практическая работа №7. От тензоров PyTorch к классификации изображений

**Дицель Н.С., ИКБО-11-23.** Проверил: ст. преподаватель Приходько Н.А. Дата: 07.11.2026.

Цель: проверить операции PyTorch и шаг обучения, обучить MNIST, сравнить линейную модель с CNN на BloodMNIST, выполнить CIFAR-10 для трёх классов.

Работа выполняется сверху вниз. Для всех классификаторов — пять эпох, seed=42; итоговый test считается с параметрами пятой эпохи, как требует методичка. Краткие выводы сопровождают расчёты. Полные вопросы и короткие ответы — в отдельном файле «Ответы_на_вопросы_7.md».""")
md('## 1. Окружение')
code(r'''
from pathlib import Path
from time import perf_counter
import json, platform, random, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from IPython.display import display

ROOT=Path.cwd()
if not (ROOT/"Практическая работа №7.docx").exists():
    ROOT=ROOT/"Практическая работа №7"
assert (ROOT/"Практическая работа №7.docx").exists(),ROOT
DATA=ROOT/"practice7_data";OUT=ROOT/"practice7_results"
DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
torch.set_num_threads(2)
torch.set_printoptions(precision=3,sci_mode=False)
device=torch.device("cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu")
if device.type=="cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
plt.rcParams.update({"font.family":"DejaVu Sans","figure.dpi":100})
environment=dict(python=platform.python_version(),torch=torch.__version__,
    torchvision=torchvision.__version__,numpy=np.__version__,device=str(device),
    cpu_threads=torch.get_num_threads(),seed=SEED)
print(environment)
(OUT/"environment.json").write_text(json.dumps(environment,indent=2),encoding="utf-8")
def savefig(name):
    plt.gcf().savefig(OUT/name,dpi=160,bbox_inches="tight")
    plt.show()
def params(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)
''')
md('## 2. Тензоры, Linear и один шаг обучения\n\nПроверяем формы, общую память NumPy, матричное умножение и градиенты на малых примерах.')
code(r'''
a=np.arange(1,7,dtype=np.float32).reshape(2,3)
x=torch.tensor(a)
assert np.allclose(a*2+1,(x*2+1).numpy())
print("Форма/оси/элементы:",x.shape,x.ndim,x.numel(),x.dtype,x.device)
print("Второй столбец:",x[:,1],"Средние столбцов:",x.mean(0),"строк:",x.mean(1))
print("Broadcasting:",x+torch.tensor([10.,20.,30.]))
a2=torch.tensor([[1.,2.],[3.,4.]]);b2=torch.tensor([[5.,6.],[7.,8.]])
print("*:",a2*b2,"@:",a2@b2)
image_hwc=torch.arange(18).reshape(2,3,3)
image_chw=image_hwc.permute(2,0,1)
batch=image_chw.unsqueeze(0)
print("HWC → CHW → NCHW:",image_hwc.shape,image_chw.shape,batch.shape)
assert not torch.equal(image_chw,image_hwc.reshape(3,2,3))
print("Flatten сохраняет пакет:",batch.flatten(1).shape)
shared=torch.from_numpy(a);copied=shared.clone();a[0,0]=100
assert shared[0,0]==100 and copied[0,0]==1
print("from_numpy делит память; clone создаёт независимую копию.")
''')
code(original(16,18,20,22)+r'''
assert torch.allclose(Y_manual,Y_layer)
assert params(layer)==8
print("Число параметров не зависит от размера пакета.")
''')
md('**Вывод:** Linear вычисляет XWᵀ+b; восемь параметров — шесть весов и два смещения. Перестановка осей через permute сохраняет каналы, reshape этого не гарантирует.')
code(original(24))
md('**Вывод:** у полносвязного слоя число весов быстро растёт с площадью изображения. Таблица оценивает только пиксели и веса float32; обучение дополнительно хранит градиенты и промежуточные значения.')
code(original(26,28,30)+r'''
assert torch.allclose(separate,combined,atol=1e-6)
print("Два слоя без активации сворачиваются в один; ReLU добавляет нелинейность.")
''')
code(original(32,34,36,38)+r'''
assert torch.allclose(regression[0].weight,w_expected)
assert torch.allclose(regression[0].bias,b_expected)
assert torch.allclose(manual_dw,torch.tensor(-22.666666),atol=1e-5)
assert new_loss.item()<loss.item()
''')
md('**Вывод:** zero_grad очищает накопленные градиенты, backward вычисляет производные, step меняет параметры. Ручной расчёт совпал с PyTorch, после шага MSE уменьшилась.')
code(original(40)+r'''
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
axes[0].plot(range(1,201),curve_losses)
axes[0].set(xlabel="Эпоха",ylabel="MSE train",title="Нелинейная регрессия: обучение")
axes[1].plot(x_curve.numpy(),y_curve.numpy(),label="x²")
axes[1].plot(x_curve.numpy(),curve_prediction.numpy(),"--",label="Модель")
axes[1].set(xlabel="x",ylabel="y",title="Обучающие точки");axes[1].legend()
plt.tight_layout();savefig("01_tensor_regression.png")
''')
code(original(45)+r'''
assert torch.allclose(manual_loss,loss)
''')
md('**Вывод:** CrossEntropyLoss получает логиты; Softmax отдельно перед ней не нужен. В примере проверено совпадение с ручной кросс-энтропией.')
md('## 3. Dataset, DataLoader и общий цикл обучения')
code(original(47,49,51))
md('**Вывод:** Dataset возвращает один объект с меткой; DataLoader собирает и перемешивает пакеты. Для valid/test перемешивание выключено, num_workers=0 подходит для Windows.')
code(r'''
def synchronize():
    if device.type=="cuda":torch.cuda.synchronize()
    elif device.type=="mps":torch.mps.synchronize()

@torch.no_grad()
def predict(model,loader):
    model.eval()
    answers,predictions,probabilities=[],[],[]
    for images,labels in loader:
        logits=model(images.to(device))
        answers.append(labels.cpu())
        predictions.append(logits.argmax(1).cpu())
        probabilities.append(logits.softmax(1).cpu())
    return (torch.cat(answers).numpy(),torch.cat(predictions).numpy(),
            torch.cat(probabilities).numpy())

def fit(model,train_data,valid_loader,name):
    # Один порядок train-пакетов для сравниваемых моделей.
    loader=DataLoader(train_data,batch_size=128,shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(SEED))
    optimizer=optim.Adam(model.parameters(),lr=.001)
    criterion=nn.CrossEntropyLoss()
    history=[]
    for epoch in range(1,6):
        model.train()
        synchronize();started=perf_counter();loss_sum=0.
        for images,labels in loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            logits=model(images)
            loss=criterion(logits,labels)
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()*len(labels)
        synchronize();train_seconds=perf_counter()-started
        answers,predictions,_=predict(model,valid_loader)
        row=dict(epoch=epoch,train_loss=loss_sum/len(train_data),
                 valid_accuracy=float(accuracy_score(answers,predictions)),
                 train_seconds=train_seconds)
        history.append(row);print(name,row,flush=True)
    frame=pd.DataFrame(history)
    frame.to_csv(OUT/f"{name}_history.csv",index=False)
    return frame

def assess(model,loader,name,class_names):
    # eval и no_grad не должны изменять веса или буферы BatchNorm.
    snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    answers,predictions,probability=predict(model,loader)
    assert all(torch.equal(snapshot[k],v.detach().cpu()) for k,v in model.state_dict().items())
    metrics=dict(accuracy=float(accuracy_score(answers,predictions)),
        macro_f1=float(f1_score(answers,predictions,labels=range(len(class_names)),
            average="macro",zero_division=0)),test_size=len(answers),parameters=params(model))
    report=pd.DataFrame(classification_report(answers,predictions,labels=range(len(class_names)),
        target_names=class_names,output_dict=True,zero_division=0)).T
    report.to_csv(OUT/f"{name}_by_class.csv")
    pd.DataFrame(dict(label=answers,prediction=predictions,confidence=probability.max(1))).to_csv(OUT/f"{name}_predictions.csv",index=False)
    (OUT/f"{name}_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},OUT/f"{name}_weights.pt")
    print(name,metrics)
    return answers,predictions,probability,metrics

def confusion_plot(answers,predictions,names,title,filename):
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(answers,predictions,
        labels=range(len(names)),display_labels=names,ax=ax,colorbar=False)
    ax.set(xlabel="Предсказанный класс",ylabel="Истинный класс",title=title)
    ax.tick_params(axis="x",rotation=30)
    plt.tight_layout();savefig(filename)
''')
md('## 4. MNIST: линейная классификация цифр\n\nИз исходного train выделяем 10 000 объектов train и 2 000 valid по seed=42. Официальные 10 000 test используются только после пяти эпох.')
code(original(53,54).replace('"practice8_data"','str(DATA)').replace('plt.show()','savefig("02_mnist_examples.png")'))
code(original(56,58)+r'''
assert not set(mnist_train.indices)&set(mnist_valid.indices)
assert params(mnist_model)==7850
np.savez_compressed(OUT/"mnist_split.npz",train=mnist_train.indices,valid=mnist_valid.indices)
''')
code(r'''
mnist_history=fit(mnist_model,mnist_train,mnist_valid_loader,"mnist")
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
axes[0].plot(mnist_history.epoch,mnist_history.train_loss,"o-")
axes[0].set(xlabel="Эпоха",ylabel="Cross-entropy train",title="MNIST: потеря")
axes[1].plot(mnist_history.epoch,mnist_history.valid_accuracy,"o-")
axes[1].set(xlabel="Эпоха",ylabel="Accuracy valid",ylim=(0,1),title="MNIST: качество")
plt.tight_layout();savefig("03_mnist_history.png")
mnist_test=datasets.MNIST(root=str(DATA),train=False,download=True,transform=transforms.ToTensor())
mnist_test_loader=DataLoader(mnist_test,batch_size=256,shuffle=False,num_workers=0)
mnist_answers,mnist_predictions,mnist_probability,mnist_metrics=assess(
    mnist_model,mnist_test_loader,"mnist",[str(i) for i in range(10)])
confusion_plot(mnist_answers,mnist_predictions,[str(i) for i in range(10)],
    "MNIST: test, линейная модель","04_mnist_confusion.png")
''')
code(original(67).replace('plt.show()','savefig("05_mnist_weights.png")'))
md('**Вывод:** одна строка весов показывает вклад пикселей в логит цифры 3. Это не вероятности пикселей; окончательный класс определяется сравнением всех десяти логитов.')
md('## 5. Свёртка, pooling и BatchNorm\n\nСвёртка PyTorch использует ядро без переворота. Вручную сверяем карту откликов, шаг, padding и суммирование входных каналов.')
code(original(69,71,73,77)+r'''
assert torch.allclose(output[0,0],manual_map)
''')
code(original(79,81,83)+r'''
assert torch.allclose(normalized,manual_normalized,atol=1e-5)
assert torch.allclose(evaluated,manual_eval,atol=1e-5)
''')
md('**Вывод:** pooling не обучает веса. Conv2d суммирует отклики всех входных каналов; одно ядро повторяется в разных позициях. BatchNorm2d в train считает статистики по N,H,W отдельно для каждого C, в eval использует накопленные оценки.')
md('## 6. BloodMNIST: линейная модель против CNN\n\nОдинаковые данные и нормировка для обеих моделей: 600 train-изображений каждого из восьми классов, всего 4 800. Официальные valid/test оставляем целиком; архитектуры, Adam и пять эпох заданы заранее.')
code(original(86,88).replace('Path("practice8_data")','DATA').replace('plt.show()','savefig("06_blood_examples.png")'))
code(original(90).replace('plt.show()','savefig("07_blood_transforms.png")'))
md('**Вывод:** перевод в серый теряет цвет, размытие — мелкие границы, уменьшение с обратным увеличением не восстанавливает детали. Для обучения оставляем RGB и нормировку каналов в диапазон примерно −1…1.')
code(original(92,94,96,98).replace('plt.show()','savefig("08_blood_parameters.png")')+r'''
assert len(blood_train)==4800
assert params(blood_linear)==18824 and params(blood_cnn)==7696
np.savez_compressed(OUT/"blood_train_indices.npz",indices=train_indices)
''')
code(r'''
linear_history=fit(blood_linear,blood_train,blood_valid_loader,"blood_linear")
cnn_history=fit(blood_cnn,blood_train,blood_valid_loader,"blood_cnn")
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
for label,history in [("Линейная",linear_history),("CNN",cnn_history)]:
    axes[0].plot(history.epoch,history.train_loss,"o-",label=label)
    axes[1].plot(history.epoch,history.valid_accuracy,"o-",label=label)
axes[0].set(xlabel="Эпоха",ylabel="Cross-entropy train",title="BloodMNIST: потеря")
axes[1].set(xlabel="Эпоха",ylabel="Accuracy valid",ylim=(0,1),title="BloodMNIST: качество")
for ax in axes:ax.legend()
plt.tight_layout();savefig("09_blood_history.png")
''')
code(original(107).replace('plt.show()','savefig("10_blood_feature_maps.png")')+r'''
assert torch.allclose(logits_from_vector,logits_direct)
''')
md('**Вывод:** первый блок CNN создаёт восемь карт; перед последним Linear изображение превращается в вектор из 784 признаков. Прямой проход через выделенный вектор совпал с полным проходом сети.')
code(original(109)+r'''
blood_predictions={};blood_rows=[]
for key,label,model,history in [
    ("blood_linear","Линейная",blood_linear,linear_history),
    ("blood_cnn","CNN",blood_cnn,cnn_history)]:
    truth,predicted,probability,metrics=assess(model,blood_test_loader,key,blood_names)
    blood_predictions[label]=predicted
    blood_rows.append({"Модель":label,"Accuracy test":metrics["accuracy"],
        "Macro-F1 test":metrics["macro_f1"],"Параметров":metrics["parameters"],
        "Обучение 5 эпох, с":float(history.train_seconds.sum())})
comparison=pd.DataFrame(blood_rows).merge(speed_table,on="Модель")
comparison.to_csv(OUT/"bloodmnist_comparison.csv",index=False)
display(comparison.round(4))
confusion_plot(blood_test_labels,blood_predictions["CNN"],[str(i) for i in range(8)],
    "BloodMNIST: test, CNN","11_blood_confusion.png")
''')
code(original(115).replace('plt.show()','savefig("12_blood_errors.png")'))
md('**Вывод:** сравниваем качество, число параметров и время отдельно. Малое число весов CNN не гарантирует быстрого применения: свёртки создают и обрабатывают карты признаков. Время относится к этому устройству, пакету 64 и исключает чтение/перенос данных.')

exec((ROOT/'_build/cifar_cells.py').read_text(encoding='utf-8'))
md('## 8. Итоговые результаты\n\nФактические метрики и краткий разбор ошибок добавляются после выполнения в «Результаты_практики_7.md» и в этот раздел. Вопросы из лекции №7 (слайды 70–71) и раздела 8 практики №7 — отдельно в «Ответы_на_вопросы_7.md».')
nb=nbf.v4.new_notebook(cells=C,metadata={"kernelspec":{"display_name":"Python 3","name":"python3","language":"python"}})
nbf.write(nb,ROOT/'Практика_7.ipynb')
print('Создано ячеек:',len(C))
