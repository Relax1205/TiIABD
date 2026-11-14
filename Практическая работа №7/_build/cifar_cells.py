"""Самостоятельная часть ПР7; вставляется сборщиком в общий блокнот."""
md("""## 7. Самостоятельная часть: CIFAR-10, вариант №1

Правило назначения по группе/зачётке в материалах не указано; использован вариант 1 из заготовки: самолёт, автомобиль, корабль. Он задаётся одной переменной variant.

Из исходного train каждого класса берём 1 500 train и следующие 300 valid после фиксированной перестановки. Из официального test отдельно берём по 300 объектов. Итого **4 500 / 900 / 900**. Используем требование методички; упоминания 100 test и общего train=1 500 в заготовке считаем опечатками.

RGB 32×32 → Resize 28×28 → ToTensor → Normalize(0,5; 0,5). Преобразование выполняется один раз в исходном CIFAR10.""")
code(r'''
variant=1
variants={1:[0,1,8],2:[2,4,6],3:[3,5,7],4:[1,8,9],5:[0,2,7]}
selected_classes=variants[variant]
all_names=["Самолёт","Автомобиль","Птица","Кошка","Олень","Собака","Лягушка","Лошадь","Корабль","Грузовик"]
class_names=[all_names[i] for i in selected_classes]
label_map={old:new for new,old in enumerate(selected_classes)}
display(pd.DataFrame({"Исходный код":selected_classes,"Новый код":range(3),"Класс":class_names}))
cifar_transform=transforms.Compose([
    transforms.Resize((28,28),antialias=True),transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
cifar_train_source=datasets.CIFAR10(root=str(DATA),train=True,download=True,transform=cifar_transform)
cifar_test_source=datasets.CIFAR10(root=str(DATA),train=False,download=True,transform=cifar_transform)
rng=np.random.default_rng(SEED)
cifar_indices={"train":[],"valid":[],"test":[]}
train_targets=np.array(cifar_train_source.targets)
test_targets=np.array(cifar_test_source.targets)
for label in selected_classes:
    candidates=rng.permutation(np.flatnonzero(train_targets==label))
    cifar_indices["train"].extend(candidates[:1500].tolist())
    cifar_indices["valid"].extend(candidates[1500:1800].tolist())
    candidates=rng.permutation(np.flatnonzero(test_targets==label))
    cifar_indices["test"].extend(candidates[:300].tolist())
assert not set(cifar_indices["train"])&set(cifar_indices["valid"])
for split,expected in [("train",4500),("valid",900),("test",900)]:
    assert len(cifar_indices[split])==len(set(cifar_indices[split]))==expected
manifest=[]
for split,indices in cifar_indices.items():
    labels=test_targets if split=="test" else train_targets
    for index in indices:
        manifest.append(dict(split=split,source_split="test" if split=="test" else "train",
            original_index=index,original_label=int(labels[index]),label=label_map[int(labels[index])]))
manifest=pd.DataFrame(manifest)
manifest.to_csv(OUT/"cifar_split.csv",index=False)
print(manifest.groupby(["split","label"]).size().unstack().to_string())
assert (manifest.groupby(["split","label"]).size().unstack().loc["train"]==1500).all()
assert (manifest.groupby(["split","label"]).size().unstack().loc[["valid","test"]]==300).all().all()
''')
md('### 7.1. Собственный Dataset и проверка пакета')
code(r'''
class CifarSubset(Dataset):
    def __init__(self,source,indices,label_map):
        self.source=source;self.indices=list(indices);self.label_map=dict(label_map)
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        image,label=self.source[self.indices[index]]
        return image,self.label_map[label]

cifar_train=CifarSubset(cifar_train_source,cifar_indices["train"],label_map)
cifar_valid=CifarSubset(cifar_train_source,cifar_indices["valid"],label_map)
cifar_test=CifarSubset(cifar_test_source,cifar_indices["test"],label_map)
cifar_train_loader=DataLoader(cifar_train,batch_size=128,shuffle=True,num_workers=0,
    generator=torch.Generator().manual_seed(SEED))
cifar_valid_loader=DataLoader(cifar_valid,batch_size=256,shuffle=False,num_workers=0)
cifar_test_loader=DataLoader(cifar_test,batch_size=256,shuffle=False,num_workers=0)
images,labels=next(iter(cifar_train_loader))
assert images.shape==(128,3,28,28) and labels.shape==(128,)
assert images.dtype==torch.float32 and labels.dtype==torch.int64
assert set(labels.tolist())<=set(range(3))
print("Пакет:",images.shape,images.dtype,"Метки:",labels.shape,labels.dtype)
fig,axes=plt.subplots(1,3,figsize=(8,3))
for k,ax in enumerate(axes):
    i=next(i for i in range(len(cifar_train)) if cifar_train[i][1]==k)
    image,label=cifar_train[i]
    ax.imshow((image*.5+.5).permute(1,2,0).clamp(0,1).numpy())
    ax.set(title=class_names[label],xlabel="Ширина",ylabel="Высота")
plt.suptitle("CIFAR-10: по одному train-примеру каждого класса")
plt.tight_layout();savefig("13_cifar_examples.png")
''')
md('### 7.2. CNN и пять эпох\n\nДва блока Conv2d → BatchNorm2d → ReLU → MaxPool2d и Linear на три логита. Размеры: 3×28×28 → 8×14×14 → 16×7×7 → 784 → 3. Архитектура и число эпох фиксируются до test.')
code(r'''
def make_cifar_model():
    return nn.Sequential(
        nn.Conv2d(3,8,3,padding=1,bias=False),nn.BatchNorm2d(8),nn.ReLU(),nn.MaxPool2d(2),
        nn.Conv2d(8,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
        nn.Flatten(),nn.Linear(16*7*7,3))
torch.manual_seed(SEED)
cifar_model=make_cifar_model().to(device)
print(cifar_model,"Параметров:",params(cifar_model))
assert params(cifar_model)==3771
cifar_model.eval();current=torch.zeros(1,3,28,28,device=device);shapes=[]
with torch.no_grad():
    for layer in cifar_model:
        current=layer(current)
        shapes.append(dict(layer=type(layer).__name__,shape=str(tuple(current.shape)),parameters=params(layer)))
display(pd.DataFrame(shapes))
with torch.no_grad():assert cifar_model(images.to(device)).shape==(len(images),3)
cifar_history=fit(cifar_model,cifar_train,cifar_valid_loader,"cifar")
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
axes[0].plot(cifar_history.epoch,cifar_history.train_loss,"o-")
axes[0].set(xlabel="Эпоха",ylabel="Cross-entropy train",title="CIFAR-10: потеря")
axes[1].plot(cifar_history.epoch,cifar_history.valid_accuracy,"o-")
axes[1].set(xlabel="Эпоха",ylabel="Accuracy valid",ylim=(0,1),title=f"Вариант {variant}: качество")
plt.tight_layout();savefig("14_cifar_history.png")
''')
md('### 7.3. Test и первые три ошибки\n\nВ итоговом проходе — eval и no_grad, без backward/step. Берём состояние пятой эпохи, а не подбираем эпоху по test.')
code(r'''
cifar_answers,cifar_predictions,cifar_probability,cifar_metrics=assess(
    cifar_model,cifar_test_loader,"cifar",class_names)
assert len(cifar_answers)==900
confusion_plot(cifar_answers,cifar_predictions,class_names,
    f"CIFAR-10: test, вариант {variant}","15_cifar_confusion.png")
cifar_error_indices=np.flatnonzero(cifar_answers!=cifar_predictions)[:3]
error_rows=[]
if len(cifar_error_indices):
    fig,axes=plt.subplots(1,len(cifar_error_indices),figsize=(10,3),squeeze=False)
    for ax,index in zip(axes.flat,cifar_error_indices):
        image,label=cifar_test[int(index)];pred=int(cifar_predictions[index])
        ax.imshow((image*.5+.5).permute(1,2,0).clamp(0,1).numpy())
        ax.set(title=f"{class_names[label]} → {class_names[pred]}",
            xlabel=f"Индекс test: {index}",ylabel="Высота")
        error_rows.append(dict(test_position=int(index),source_index=cifar_indices["test"][int(index)],
            label=int(label),prediction=pred,confidence=float(cifar_probability[index,pred])))
    plt.suptitle("Первые ошибки: истинный класс → ответ CNN")
    plt.tight_layout();savefig("16_cifar_errors.png")
else:print("Ошибок на test нет.")
pd.DataFrame(error_rows).to_csv(OUT/"cifar_error_examples.csv",index=False)
restored=make_cifar_model().to(device)
restored.load_state_dict(torch.load(OUT/"cifar_weights.pt",map_location=device,weights_only=True))
cifar_model.eval();restored.eval()
probe=next(iter(cifar_test_loader))[0][:16].to(device)
with torch.no_grad():assert torch.allclose(cifar_model(probe),restored(probe),atol=1e-6)
config=dict(variant=variant,original_classes=selected_classes,class_names=class_names,
    seed=SEED,split_sizes={k:len(v) for k,v in cifar_indices.items()},
    per_class=dict(train=1500,valid=300,test=300),image_shape=[3,28,28],
    transform="Resize 28x28, ToTensor, Normalize(mean=.5,std=.5) per channel",
    architecture="Conv3-8-BN-ReLU-Pool2, Conv8-16-BN-ReLU-Pool2, Flatten, Linear784-3",
    parameters=params(cifar_model),optimizer="Adam",lr=.001,batch_size=128,epochs=5,
    final_epoch=5,environment=environment,test=cifar_metrics,
    reload_logits_match=True)
(OUT/"cifar_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
print("Сохранены веса, параметры опыта, индексы и метрики. После загрузки логиты совпали.")
''')
md('**Вывод:** нормировка для рисунка обращается выражением image×0,5+0,5; модель получает нормированный тензор. Признаки и правила предобработки должны совпадать при повторном применении сохранённых весов. Метрики и конкретные ошибки приведены ниже после расчёта.')
