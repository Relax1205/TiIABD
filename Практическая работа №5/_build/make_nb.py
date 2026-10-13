"""Сборка блокнота Практики 5. Запуск: python _build/make_nb.py (из каталога работы)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))
TEXT = {}
for _t in sorted(Path(__file__).parent.glob("texts*.md")):
    key = None
    for line in _t.read_text(encoding="utf-8").splitlines():
        if line.startswith("@@"):
            key = line[2:].strip(); TEXT[key] = []
        elif key:
            TEXT[key].append(line)
TEXT = {k: "\n".join(v).strip() for k, v in TEXT.items()}
def txt(key): md(TEXT.get(key, f"*[{key}]*"))

# ---------------------------------------------------------------- титул
md("""
# Практическая работа №5. Машинное обучение в анализе данных с использованием Spark MLlib

| | |
|---|---|
| **Дисциплина** | Технологии и инструментарий анализа больших данных |
| **Студент** | Дицель Н.С. |
| **Группа** | ИКБО-11-23 |
| **Проверил** | старший преподаватель Приходько Н.А. |
| **Дата выполнения** | 13.10.2026 |

**Содержание**

0. Окружение и SparkSession
1. Чтение и контроль исходных данных (разд. 3 методички)
2. Классификация покупательского намерения — Online Shoppers (разд. 4)
3. Регрессия спроса — Seoul Bike (разд. 5)
4. Кластеризация интернет-сессий — Online Shoppers (разд. 6)
5. Сохранение и повторное применение моделей (разд. 7)
6. Самостоятельная работа — регрессия времени прослушивания подкастов (разд. 8)
7. Ответы на контрольные вопросы (разд. 9)
8. Общие выводы

Блокнот выполняется сверху вниз после перезапуска ядра (Restart & Run All). Все модели обучаются в Spark MLlib;
pandas / NumPy / Matplotlib / scikit-learn используются только для небольших итоговых таблиц, кривых и графиков.
""")

# ---------------------------------------------------------------- 0
md("""
## 0. Окружение и SparkSession

В методичке сессия считается уже созданной, поэтому здесь она создаётся явно в режиме `local[*]`.
На Windows для записи моделей (`PipelineModel.save`) Spark использует Hadoop FileSystem, поэтому подключаются `winutils.exe` и `hadoop.dll`.
""")
code(r'''
import os, sys, platform, subprocess
if os.name == "nt" and os.path.isdir(r"C:\hadoop\bin"):
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pyspark
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .master("local[*]")
    .appName("Practice5_MLlib")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate())

java = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
print("ОС     :", platform.platform())
print("Python :", sys.version.split()[0])
print("Java   :", java)
print("PySpark:", pyspark.__version__)
print("CPU    :", os.cpu_count())
''')

# ---------------------------------------------------------------- 3
md("""
## 1. Чтение и контроль исходных данных

Исходные файлы `online_shoppers_intention.csv` и `SeoulBikeData.csv` скачаны с UCI Machine Learning Repository
(DOI 10.24432/C5F88Q и 10.24432/C5F62R) и лежат рядом с блокнотом.

### Практический блок 3.1. Импорты и рабочие каталоги
""")
code(r'''
from pathlib import Path
from time import perf_counter
import csv
import json
import math
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from pyspark.sql import functions as F, Window
from pyspark.sql.types import (
    StructType, StructField, LongType, IntegerType,
    DoubleType, StringType, BooleanType
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import (
    SQLTransformer, StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler, PolynomialExpansion, PCA, Imputer
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, RegressionEvaluator, ClusteringEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    silhouette_score
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# Единый стиль графиков: фиксированные цвета сущностей, приглушённая сетка
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # модель 1 / модель 2 / модель 3
GRAY = "#8a8985"                               # базовые правила и вспомогательные линии
PALETTE = [C1, C2, C3, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
})

SEED = 42
ROOT = Path.cwd()
WORK = ROOT / "practice5_data"
OUT = ROOT / "practice5_results"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SHOP_SOURCE = ROOT / "online_shoppers_intention.csv"
BIKE_SOURCE = ROOT / "SeoulBikeData.csv"

for path in [SHOP_SOURCE, BIKE_SOURCE]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл: {path}")

sc = spark.sparkContext
sc.setLogLevel("WARN")
print("Spark:", spark.version)
print("Master:", sc.master)
print("Spark UI:", sc.uiWebUrl)
''')

md("""
Переменная `sc` указывает на работающий SparkContext. Адрес Spark UI выводится фактически: порт может отличаться от 4040.

### Практический блок 3.2. Устойчивый row_id для интернет-сессий

`row_id` повторяет исходный порядок CSV и не входит в features. Он нужен для воспроизводимого разбиения,
сохранения предсказаний и соединения кластерных меток с исходной таблицей.
""")
code(r'''
SHOP_WORK = WORK / "online_shoppers_with_id.csv"

with SHOP_SOURCE.open("r", encoding="utf-8-sig", newline="") as src, \
     SHOP_WORK.open("w", encoding="utf-8", newline="") as dst:
    reader = csv.reader(src)
    writer = csv.writer(dst)
    header = next(reader)
    writer.writerow(["row_id"] + header)
    for row_id, row in enumerate(reader, start=1):
        writer.writerow([row_id] + row)

print("Исходный заголовок:", header)
''')

md("### Практический блок 3.3. Явные схемы и чтение CSV")
code(r'''
shop_schema = StructType([
    StructField("row_id", LongType(), False),
    StructField("Administrative", DoubleType(), False),
    StructField("Administrative_Duration", DoubleType(), False),
    StructField("Informational", DoubleType(), False),
    StructField("Informational_Duration", DoubleType(), False),
    StructField("ProductRelated", DoubleType(), False),
    StructField("ProductRelated_Duration", DoubleType(), False),
    StructField("BounceRates", DoubleType(), False),
    StructField("ExitRates", DoubleType(), False),
    StructField("PageValues", DoubleType(), False),
    StructField("SpecialDay", DoubleType(), False),
    StructField("Month", StringType(), False),
    StructField("OperatingSystems", IntegerType(), False),
    StructField("Browser", IntegerType(), False),
    StructField("Region", IntegerType(), False),
    StructField("TrafficType", IntegerType(), False),
    StructField("VisitorType", StringType(), False),
    StructField("Weekend", BooleanType(), False),
    StructField("Revenue", BooleanType(), False),
])

bike_schema = StructType([
    StructField("date_text", StringType(), False),
    StructField("label", DoubleType(), False),
    StructField("hour", IntegerType(), False),
    StructField("temperature", DoubleType(), False),
    StructField("humidity", DoubleType(), False),
    StructField("wind_speed", DoubleType(), False),
    StructField("visibility", DoubleType(), False),
    StructField("dew_point", DoubleType(), False),
    StructField("solar_radiation", DoubleType(), False),
    StructField("rainfall", DoubleType(), False),
    StructField("snowfall", DoubleType(), False),
    StructField("season", StringType(), False),
    StructField("holiday", StringType(), False),
    StructField("functioning_day", StringType(), False),
])

shop_raw = (spark.read
    .option("header", True)
    .option("mode", "FAILFAST")
    .schema(shop_schema)
    .csv(SHOP_WORK.resolve().as_posix()))

bike_raw = (spark.read
    .option("header", True)
    .option("encoding", "ISO-8859-1")
    .option("enforceSchema", True)
    .option("mode", "FAILFAST")
    .schema(bike_schema)
    .csv(BIKE_SOURCE.resolve().as_posix()))

# Контроль порядка столбцов Seoul Bike: исходный заголовок против назначенных имён
with BIKE_SOURCE.open("r", encoding="ISO-8859-1") as f:
    bike_header = next(csv.reader(f))
for original, safe in zip(bike_header, bike_schema.fieldNames()):
    print(f"{original:30s} -> {safe}")

print("\nOnline Shoppers:", shop_raw.count(), "строк")
print("Seoul Bike:", bike_raw.count(), "строк")
shop_raw.printSchema()
bike_raw.printSchema()
''')

md("""
Файл `SeoulBikeData.csv` использует однобайтовую кодировку для символа градуса в заголовке, поэтому чтение выполняется
с `encoding="ISO-8859-1"`. Безопасные имена схемы назначаются по порядку столбцов — выше выведено соответствие исходного заголовка и новых имён.

### Практический блок 3.4. Контроль строк, ключей и пропусков
""")
code(r'''
def null_counts(df):
    return df.select(*[
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]).first().asDict()

print("Пропуски Online Shoppers:", {k: v for k, v in null_counts(shop_raw).items() if v} or "нет")
print("Пропуски Seoul Bike     :", {k: v for k, v in null_counts(bike_raw).items() if v} or "нет")
print("Уникальных row_id:", shop_raw.select("row_id").distinct().count(), "из", shop_raw.count())
print("Полных дублей сессий (без row_id):",
      shop_raw.count() - shop_raw.drop("row_id").distinct().count())
print("Уникальных пар (дата, час) в Seoul Bike:",
      bike_raw.select("date_text", "hour").distinct().count())

shop_raw.groupBy("Revenue").count().orderBy("Revenue").show()
bike_raw.groupBy("functioning_day").agg(
    F.count("*").alias("n"),
    F.min("label").alias("min_target"),
    F.max("label").alias("max_target")
).show()
''')
txt("3.4")

# ---------------------------------------------------------------- 4
md("""
## 2. Классификация покупательского намерения (Online Shoppers)

### 4.1. Постановка и дисбаланс классов

Цель — оценить вероятность `Revenue = True`. Положительный класс редкий, поэтому модели сравниваются прежде всего по PR-AUC,
precision, recall и F1, а не только по accuracy.
""")
code(r'''
cls = shop_raw.groupBy("Revenue").count().orderBy("Revenue").toPandas()
total = cls["count"].sum()
fig, ax = plt.subplots(figsize=(5.6, 3.8))
bars = ax.bar(["Без покупки", "Покупка"], cls["count"], color=[GRAY, C1], width=0.55)
for b, n in zip(bars, cls["count"]):
    ax.text(b.get_x() + b.get_width() / 2, n + 120, f"{n:,} ({n / total:.1%})".replace(",", " "),
            ha="center", va="bottom", fontsize=10)
ax.set(ylabel="Число сессий", title="Рис. 4. Распределение классов Online Shoppers",
       ylim=(0, cls["count"].max() * 1.12))
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.savefig(OUT / "classification_classes.png", dpi=160)
plt.show()
print(f"Accuracy постоянного ответа «нет покупки»: {cls['count'].iloc[0] / total:.3f}, recall покупки = 0")
''')

md("""
### 4.2–4.3. Момент доступности признаков, разбиение и веса классов

`PageValues` включён в основной опыт (оценка по накопленному состоянию сессии), а затем проверяется абляцией.
Шесть счётчиков и длительностей логарифмируются внутри Pipeline; `OperatingSystems`, `Browser`, `Region`, `TrafficType`
рассматриваются как категории. Разбиение 60/20/20 выполняется отдельно внутри каждого класса по SHA-256 от `row_id` и seed,
веса классов $w_c = N / (K\\cdot N_c)$ считаются только по train.
""")
code(r'''
SHOP_LOG_SOURCE = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration"
]
SHOP_LOG = ["log_" + c for c in SHOP_LOG_SOURCE]
SHOP_NUM_BASE = SHOP_LOG + ["BounceRates", "ExitRates", "SpecialDay"]
SHOP_CAT = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend"
]

shop = shop_raw.withColumn("label", F.col("Revenue").cast("double"))
for c in ["OperatingSystems", "Browser", "Region", "TrafficType", "Weekend"]:
    shop = shop.withColumn(c, F.col(c).cast("string"))

order_key = F.sha2(F.concat_ws(
    ":", F.col("row_id").cast("string"), F.lit(str(SEED))
), 256)
rank_window = Window.partitionBy("label").orderBy(order_key, "row_id")
class_window = Window.partitionBy("label")

shop_split = (shop
    .withColumn("_rank", F.row_number().over(rank_window))
    .withColumn("_size", F.count("*").over(class_window))
    .withColumn(
        "split",
        F.when(F.col("_rank") <= F.floor(F.col("_size") * 0.60), "train")
         .when(F.col("_rank") <= F.floor(F.col("_size") * 0.80), "valid")
         .otherwise("test")
    )
    .drop("_rank", "_size")
    .cache())

split_table = (shop_split.groupBy("split").pivot("label", [0.0, 1.0]).count()
    .withColumnRenamed("0.0", "no_purchase").withColumnRenamed("1.0", "purchase")
    .withColumn("total", F.col("no_purchase") + F.col("purchase"))
    .withColumn("purchase_share", F.round(F.col("purchase") / F.col("total"), 4))
    .orderBy(F.expr("CASE split WHEN 'train' THEN 0 WHEN 'valid' THEN 1 ELSE 2 END")))
split_table.show()

shop_train = shop_split.filter("split = 'train'").drop("split")
shop_valid = shop_split.filter("split = 'valid'").drop("split")
shop_test = shop_split.filter("split = 'test'").drop("split")

counts = {int(r["label"]): r["count"]
          for r in shop_train.groupBy("label").count().collect()}
train_n = sum(counts.values())
weight_0 = train_n / (2.0 * counts[0])
weight_1 = train_n / (2.0 * counts[1])
shop_train = shop_train.withColumn(
    "class_weight",
    F.when(F.col("label") == 1.0, F.lit(weight_1)).otherwise(F.lit(weight_0))
)
print("Веса классов: w0 = %.4f, w1 = %.4f" % (weight_0, weight_1))
''')
txt("4.3")

md("""
### Практический блок 4.5. Pipeline двух классификаторов

`SQLTransformer` создаёт логарифмированные поля; `StringIndexer` обучает словари категорий на train; `OneHotEncoder` превращает индексы
в индикаторы; `VectorAssembler` объединяет координаты. В ветке LR вектор масштабируется (`withMean=False`, чтобы не уплотнять
разреженный вектор) и передаётся в `LogisticRegression` с L2 и весами классов; в ветке RF масштабирование не нужно.
""")
code(r'''
def shop_pipeline(kind, include_page_values=True):
    expressions = [
        f"log1p(`{source}`) AS `{target}`"
        for source, target in zip(SHOP_LOG_SOURCE, SHOP_LOG)
    ]
    log_stage = SQLTransformer(
        statement="SELECT *, " + ", ".join(expressions) + " FROM __THIS__"
    )

    index_cols = [c + "_idx" for c in SHOP_CAT]
    encoded_cols = [c + "_ohe" for c in SHOP_CAT]
    indexer = StringIndexer(
        inputCols=SHOP_CAT,
        outputCols=index_cols,
        handleInvalid="keep",
        stringOrderType="alphabetAsc"
    )
    encoder = OneHotEncoder(
        inputCols=index_cols,
        outputCols=encoded_cols,
        handleInvalid="keep",
        dropLast=False
    )

    numeric = SHOP_NUM_BASE + (["PageValues"] if include_page_values else [])
    if kind == "LR":
        assembler = VectorAssembler(
            inputCols=numeric + encoded_cols,
            outputCol="features_raw"
        )
        scaler = StandardScaler(
            inputCol="features_raw", outputCol="features",
            withMean=False, withStd=True
        )
        estimator = LogisticRegression(
            maxIter=100, regParam=0.05, elasticNetParam=0.0,
            standardization=False, weightCol="class_weight"
        )
        stages = [log_stage, indexer, encoder, assembler, scaler, estimator]
    elif kind == "RF":
        assembler = VectorAssembler(
            inputCols=numeric + encoded_cols,
            outputCol="features"
        )
        estimator = RandomForestClassifier(
            numTrees=80, maxDepth=8, maxBins=64,
            featureSubsetStrategy="sqrt", seed=SEED,
            weightCol="class_weight"
        )
        stages = [log_stage, indexer, encoder, assembler, estimator]
    else:
        raise ValueError(kind)
    return Pipeline(stages=stages)


def binary_scores(model, df):
    return (model.transform(df)
        .withColumn("score", vector_to_array("probability")[1])
        .select("row_id", "label", "score"))


def binary_metrics(pred, threshold=0.5, ranking=True):
    decision = F.col("score") >= F.lit(float(threshold))
    y = F.col("label") == 1.0
    values = pred.agg(
        F.sum((decision & y).cast("long")).alias("tp"),
        F.sum((decision & ~y).cast("long")).alias("fp"),
        F.sum((~decision & y).cast("long")).alias("fn"),
        F.sum((~decision & ~y).cast("long")).alias("tn")
    ).first().asDict()

    tp, fp, fn, tn = (values[k] for k in ["tp", "fp", "fn", "tn"])
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }
    if ranking:
        for metric in ["areaUnderROC", "areaUnderPR"]:
            result[metric] = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="score", metricName=metric
            ).evaluate(pred)
    return result


def show_metrics(title, m):
    print(f"{title}: accuracy={m['accuracy']:.3f}; precision={m['precision']:.3f}; "
          f"recall={m['recall']:.3f}; F1={m['f1']:.3f}"
          + (f"; ROC-AUC={m['areaUnderROC']:.3f}; PR-AUC={m['areaUnderPR']:.3f}" if "areaUnderPR" in m else ""))
    print(f"   Матрица ошибок: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}")
''')

md("""
### 4.6. Метрики и обучение

Обе полные цепочки обучаются на одном `shop_train`; `setJobGroup` даёт понятные подписи в Spark UI, `perf_counter` измеряет время `fit`.
Победитель выбирается по `areaUnderPR` на valid, а не по test.
""")
code(r'''
shop_models = {}
shop_valid_predictions = {}
shop_rows = []

for name, kind in [("Логистическая регрессия", "LR"),
                   ("Случайный лес", "RF")]:
    sc.setJobGroup("P5_CLASS_" + kind, "Классификация: " + name)
    t0 = perf_counter()
    model = shop_pipeline(kind, include_page_values=True).fit(shop_train)
    fit_s = perf_counter() - t0
    pred = binary_scores(model, shop_valid).cache()
    pred.count()
    row = {"model": name, "fit_s": fit_s, **binary_metrics(pred)}
    shop_models[name] = model
    shop_valid_predictions[name] = pred
    shop_rows.append(row)

shop_validation = pd.DataFrame(shop_rows).sort_values("areaUnderPR", ascending=False)
print("Таблица 5. Результат на valid при пороге 0,5")
print(shop_validation.round(4).to_string(index=False))
BEST_SHOP_NAME = str(shop_validation.iloc[0]["model"])
BEST_SHOP = shop_models[BEST_SHOP_NAME]
print("\nЛучшая модель по PR-AUC на valid:", BEST_SHOP_NAME)
''')
txt("4.6")

md("""
### 4.7. Выбор порога и однократный test

Порог выбирается только на valid — в максимуме F1 по `precision_recall_curve`. Таблица valid выбранной модели (≈2,5 тыс. строк, два столбца)
безопасно переносится в pandas. После фиксации порога test используется один раз.
""")
code(r'''
valid_pd = shop_valid_predictions[BEST_SHOP_NAME].toPandas()
precision, recall, thresholds = precision_recall_curve(
    valid_pd["label"], valid_pd["score"]
)
f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
best_index = int(np.nanargmax(f1_values[:-1]))
SHOP_THRESHOLD = float(thresholds[best_index])
print("Выбранная модель:", BEST_SHOP_NAME)
print("Порог по максимуму F1 на valid:", round(SHOP_THRESHOLD, 4))
shop_valid_best = binary_metrics(shop_valid_predictions[BEST_SHOP_NAME], SHOP_THRESHOLD, ranking=True)
show_metrics("VALID (выбранный порог)", shop_valid_best)

sc.setJobGroup("P5_CLASS_TEST", "Классификация: однократная проверка на test")
shop_test_scores = binary_scores(BEST_SHOP, shop_test).cache()
shop_test_scores.count()
shop_test_metrics = binary_metrics(shop_test_scores, SHOP_THRESHOLD, ranking=True)
show_metrics("TEST", shop_test_metrics)
''')
txt("4.7")

md("""
### 4.8. Кривые качества

Кривые строятся по небольшим локальным таблицам `score`/`label`, обучение и предсказания остаются в Spark.
Дополнительно к методичке построен график precision / recall / F1 в зависимости от порога — он показывает, где именно был зафиксирован порог.
""")
code(r'''
MODEL_COLORS = {"Логистическая регрессия": C1, "Случайный лес": C2}
local_valid = {name: shop_valid_predictions[name].toPandas() for name in MODEL_COLORS}

# Рис. 8. Precision–Recall
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for name, color in MODEL_COLORS.items():
    p, r, _ = precision_recall_curve(local_valid[name]["label"], local_valid[name]["score"])
    ax.plot(r, p, label=name, color=color, linewidth=2)
ax.axhline(valid_pd["label"].mean(), linestyle="--", color=GRAY,
           label="Доля положительного класса")
ax.set(xlabel="Recall", ylabel="Precision", title="Рис. 8. Precision–Recall-кривые на valid")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "classification_pr.png", dpi=160); plt.show()

# Рис. 9. ROC
fig, ax = plt.subplots(figsize=(6.2, 5.2))
for name, color in MODEL_COLORS.items():
    fpr, tpr, _ = roc_curve(local_valid[name]["label"], local_valid[name]["score"])
    ax.plot(fpr, tpr, label=name, color=color, linewidth=2)
ax.plot([0, 1], [0, 1], "--", color=GRAY, label="Случайное ранжирование")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Рис. 9. ROC-кривые на valid")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "classification_roc.png", dpi=160); plt.show()

# Дополнительно: метрики в зависимости от порога (valid, выбранная модель)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(thresholds, precision[:-1], color=C1, linewidth=2, label="Precision")
ax.plot(thresholds, recall[:-1], color=C2, linewidth=2, label="Recall")
ax.plot(thresholds, f1_values[:-1], color=C3, linewidth=2, label="F1")
ax.axvline(SHOP_THRESHOLD, color=GRAY, linestyle="--")
ax.text(SHOP_THRESHOLD + 0.01, 0.05, f"порог = {SHOP_THRESHOLD:.3f}", color="#52514e")
ax.set(xlabel="Порог score", ylabel="Значение метрики", ylim=(0, 1.02),
       title=f"Метрики valid в зависимости от порога ({BEST_SHOP_NAME})")
ax.legend(loc="center left")
plt.tight_layout(); plt.savefig(OUT / "classification_threshold.png", dpi=160); plt.show()

# Рис. 7. Матрица ошибок на test
final_pd = shop_test_scores.toPandas()
final_pred = (final_pd["score"] >= SHOP_THRESHOLD).astype(int)
cm = confusion_matrix(final_pd["label"].astype(int), final_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
ax.set_xticks([0, 1], ["Нет покупки", "Покупка"])
ax.set_yticks([0, 1], ["Нет покупки", "Покупка"])
ax.grid(False)
ax.set(xlabel="Решение модели", ylabel="Факт", title="Рис. 7. Матрица ошибок на test")
fig.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig(OUT / "classification_confusion.png", dpi=160); plt.show()
''')
txt("4.8")

md("""
### 4.9. Важность признаков, коэффициенты LR и абляция PageValues

Имена координат подготовленного вектора берутся из метаданных `VectorAssembler` (`ml_attr`), поэтому каждая one-hot-координата
получает читаемое имя вида `Month_ohe_Nov`.
""")
code(r'''
def vector_names(model, df, col):
    attrs = model.transform(df.limit(1)).schema[col].metadata["ml_attr"]["attrs"]
    names = {}
    for group in attrs.values():
        for a in group:
            names[a["idx"]] = a["name"]
    return [names[i] for i in range(len(names))]

rf_shop = shop_models["Случайный лес"]
rf_names = vector_names(rf_shop, shop_valid, "features")
rf_imp = (pd.Series(rf_shop.stages[-1].featureImportances.toArray(), index=rf_names)
          .sort_values(ascending=False))
top = rf_imp.head(15)[::-1]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh(top.index, top.values, color=C2, height=0.6)
ax.set(xlabel="Impurity-based importance", title="Рис. 10. Наиболее важные координаты случайного леса")
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(OUT / "classification_importance.png", dpi=160); plt.show()
print(rf_imp.head(10).round(4).to_string())

lr_shop = shop_models["Логистическая регрессия"]
lr_names = vector_names(lr_shop, shop_valid, "features_raw")
lr_coef = pd.Series(lr_shop.stages[-1].coefficients.toArray(), index=lr_names)
print("\nLR: 8 наибольших и 8 наименьших коэффициентов (в масштабированных единицах):")
print(pd.concat([lr_coef.sort_values(ascending=False).head(8),
                 lr_coef.sort_values().head(8)]).round(3).to_string())
''')
code(r'''
# Проверка чувствительности к PageValues на той же valid и с теми же параметрами леса
sc.setJobGroup("P5_CLASS_ABLATION", "Классификация: RF без PageValues")
rf_without_page_values = shop_pipeline("RF", include_page_values=False).fit(shop_train)
without_scores = binary_scores(rf_without_page_values, shop_valid).cache()
with_m = binary_metrics(shop_valid_predictions["Случайный лес"], ranking=True)
without_m = binary_metrics(without_scores, ranking=True)
ablation = pd.DataFrame([
    {"Вариант": "С PageValues", "ROC-AUC valid": with_m["areaUnderROC"], "PR-AUC valid": with_m["areaUnderPR"], "F1@0.5": with_m["f1"]},
    {"Вариант": "Без PageValues", "ROC-AUC valid": without_m["areaUnderROC"], "PR-AUC valid": without_m["areaUnderPR"], "F1@0.5": without_m["f1"]},
])
print("Таблица 6. Влияние признака PageValues для случайного леса")
print(ablation.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 4))
x = np.arange(2); w = 0.36
for i, (label, color) in enumerate([("С PageValues", C2), ("Без PageValues", GRAY)]):
    vals = ablation.loc[ablation["Вариант"] == label, ["ROC-AUC valid", "PR-AUC valid"]].values[0]
    bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=color, label=label)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(x, ["ROC-AUC", "PR-AUC"])
ax.set(ylim=(0, 1.05), ylabel="Значение на valid", title="Рис. 11. Качество RF до и после исключения PageValues")
ax.legend(); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "classification_ablation.png", dpi=160); plt.show()
''')
txt("4.9")

# ---------------------------------------------------------------- 5
md("""
## 3. Регрессия спроса Seoul Bike

### 5.1–5.2. Постановка, временные признаки и части выборки

Цель — число аренд за час при работающем сервисе. Часы `Functioning Day = No` исключаются до обучения. Разбиение хронологическое:
до августа 2018 — train, август–сентябрь — valid, октябрь–ноябрь — test. Час и месяц кодируются парой sin/cos, `day_index` — число дней от начала набора.
""")
code(r'''
bike = (bike_raw
    .withColumn("date", F.to_date("date_text", "dd/MM/yyyy"))
    .drop("date_text")
    .filter(F.col("functioning_day") == "Yes")
    .withColumn("weekday", F.dayofweek("date").cast("string"))
    .withColumn("hour_sin",
                F.sin(F.lit(2.0 * math.pi) * F.col("hour") / F.lit(24.0)))
    .withColumn("hour_cos",
                F.cos(F.lit(2.0 * math.pi) * F.col("hour") / F.lit(24.0)))
    .withColumn("month_sin",
                F.sin(F.lit(2.0 * math.pi) * F.month("date") / F.lit(12.0)))
    .withColumn("month_cos",
                F.cos(F.lit(2.0 * math.pi) * F.month("date") / F.lit(12.0)))
    .withColumn("day_index",
                F.datediff("date", F.lit("2017-12-01").cast("date")).cast("double")))

bike_train = bike.filter("date < DATE '2018-08-01'").cache()
bike_valid = bike.filter(
    "date >= DATE '2018-08-01' AND date < DATE '2018-10-01'"
).cache()
bike_test = bike.filter("date >= DATE '2018-10-01'").cache()

rows = []
for name, df in [("train", bike_train), ("valid", bike_valid), ("test", bike_test)]:
    r = df.agg(F.count("*").alias("n"), F.min("date").alias("from"), F.max("date").alias("to"),
               F.avg("label").alias("mean_target")).first()
    rows.append({"Часть": name, "С": r["from"], "По": r["to"], "Строк": r["n"], "Средний спрос": round(r["mean_target"], 1)})
print("Таблица 7. Временное разбиение Seoul Bike после исключения закрытых часов")
print(pd.DataFrame(rows).to_string(index=False))
assert sum(r["Строк"] for r in rows) == bike.count(), "Потеряны строки при разбиении"
print("Рабочих часов всего:", bike.count())
''')
txt("5.2")

md("""
### Практический блок 5.4. Pipeline регрессоров

Линейная ветка: индикаторы категорий + погодные и циклические координаты → `StandardScaler` → `LinearRegression` (L2).
Полиномиальная ветка: `PolynomialExpansion(degree=2)` для пяти координат (температура, влажность, радиация, hour_sin, hour_cos — 20 слагаемых)
и более сильная регуляризация. Лес: исходный числовой набор + `day_index`, без масштабирования.
""")
code(r'''
BIKE_CAT = ["season", "holiday", "weekday"]
BIKE_NUM = [
    "temperature", "humidity", "wind_speed", "visibility",
    "dew_point", "solar_radiation", "rainfall", "snowfall",
    "hour_sin", "hour_cos", "month_sin", "month_cos"
]
BIKE_RF_NUM = BIKE_NUM + ["day_index"]
POLY_SOURCE = ["temperature", "humidity", "solar_radiation", "hour_sin", "hour_cos"]
POLY_REST = [c for c in BIKE_NUM if c not in POLY_SOURCE]


def bike_pipeline(kind):
    index_cols = [c + "_idx" for c in BIKE_CAT]
    encoded_cols = [c + "_ohe" for c in BIKE_CAT]
    indexer = StringIndexer(
        inputCols=BIKE_CAT, outputCols=index_cols,
        handleInvalid="keep", stringOrderType="alphabetAsc"
    )
    encoder = OneHotEncoder(
        inputCols=index_cols, outputCols=encoded_cols,
        handleInvalid="keep", dropLast=False
    )
    stages = [indexer, encoder]

    if kind == "LINEAR":
        stages += [
            VectorAssembler(inputCols=BIKE_NUM + encoded_cols, outputCol="features_raw"),
            StandardScaler(inputCol="features_raw", outputCol="features",
                           withMean=False, withStd=True),
            LinearRegression(maxIter=150, regParam=0.1, elasticNetParam=0.0,
                             standardization=False)
        ]
    elif kind == "POLY":
        stages += [
            VectorAssembler(inputCols=POLY_SOURCE, outputCol="poly_input"),
            PolynomialExpansion(degree=2, inputCol="poly_input", outputCol="poly_features"),
            VectorAssembler(inputCols=["poly_features"] + POLY_REST + encoded_cols,
                            outputCol="features_raw"),
            StandardScaler(inputCol="features_raw", outputCol="features",
                           withMean=False, withStd=True),
            LinearRegression(maxIter=200, regParam=1.0, elasticNetParam=0.0,
                             standardization=False)
        ]
    elif kind == "RF":
        stages += [
            VectorAssembler(inputCols=BIKE_RF_NUM + encoded_cols, outputCol="features"),
            RandomForestRegressor(
                numTrees=120, maxDepth=12, maxBins=64,
                minInstancesPerNode=5, featureSubsetStrategy="0.7",
                seed=SEED
            )
        ]
    else:
        raise ValueError(kind)
    return Pipeline(stages=stages)
''')

md("""
### 5.5. Метрики и базовые правила

`regression_metrics` вызывает `RegressionEvaluator` для RMSE, MAE, R² и считает долю отрицательных прогнозов.
Базовые правила: среднее train и среднее по часу суток (тоже только по train).
""")
code(r'''
def regression_metrics(pred, label="label", prediction="prediction"):
    result = {}
    for metric in ["rmse", "mae", "r2"]:
        result[metric] = RegressionEvaluator(
            labelCol=label, predictionCol=prediction, metricName=metric
        ).evaluate(pred)
    result["negative_share"] = pred.agg(
        F.avg((F.col(prediction) < 0).cast("double"))
    ).first()[0]
    return result

train_mean = bike_train.agg(F.avg("label")).first()[0]
hour_means = bike_train.groupBy("hour").agg(
    F.avg("label").alias("hour_mean")
)

def bike_baseline(df, by_hour=False):
    if not by_hour:
        return df.withColumn("prediction", F.lit(train_mean))
    return (df.join(hour_means, "hour", "left")
        .withColumn("prediction", F.coalesce("hour_mean", F.lit(train_mean))))

print("Среднее train:", round(train_mean, 1))
''')

md("""
### 5.6. Обучение и сравнение

Три Pipeline обучаются на `bike_train`; лучшая обучаемая модель выбирается по минимальной RMSE valid и только затем применяется к test.
После фиксации выбора для наглядности (рис. 7) на test считаются и остальные решения — на выбор это уже не влияет.
""")
code(r'''
bike_models = {}
bike_rows = []
bike_valid_predictions = {}

for name, kind in [("Линейная регрессия", "LINEAR"),
                   ("Полиномиальная регрессия", "POLY"),
                   ("Случайный лес", "RF")]:
    sc.setJobGroup("P5_REG_" + kind, "Регрессия: " + name)
    t0 = perf_counter()
    model = bike_pipeline(kind).fit(bike_train)
    fit_s = perf_counter() - t0
    pred = model.transform(bike_valid).cache()
    pred.select(F.avg("prediction")).first()
    bike_models[name] = model
    bike_valid_predictions[name] = pred
    bike_rows.append({"model": name, "fit_s": fit_s, **regression_metrics(pred)})

for name, by_hour in [("Среднее train", False), ("Среднее по часу", True)]:
    bike_rows.append({"model": name, **regression_metrics(bike_baseline(bike_valid, by_hour))})

bike_validation = pd.DataFrame(bike_rows).sort_values("rmse")
print("Метрики на valid:")
print(bike_validation.round(3).to_string(index=False))

learned = bike_validation[bike_validation["model"].isin(bike_models)]
BEST_BIKE_NAME = str(learned.iloc[0]["model"])
BEST_BIKE = bike_models[BEST_BIKE_NAME]
print("\nВыбрана модель:", BEST_BIKE_NAME)

sc.setJobGroup("P5_REG_TEST", "Регрессия: однократная проверка на test")
bike_test_prediction = (BEST_BIKE.transform(bike_test)
    .select("date", "hour", "label", "prediction")
    .withColumn("residual", F.col("label") - F.col("prediction"))
    .cache())
bike_test_metrics = regression_metrics(bike_test_prediction)
print("TEST:", {k: round(v, 3) for k, v in bike_test_metrics.items()})
''')
code(r'''
# Таблица 8 и рис. 7: справочный пересчёт на test всех решений после фиксации выбора
test_rows = {}
for name, model in bike_models.items():
    test_rows[name] = regression_metrics(model.transform(bike_test))
for name, by_hour in [("Среднее train", False), ("Среднее по часу", True)]:
    test_rows[name] = regression_metrics(bike_baseline(bike_test, by_hour))

order = ["Среднее train", "Среднее по часу", "Линейная регрессия", "Полиномиальная регрессия", "Случайный лес"]
v = bike_validation.set_index("model")
table8 = pd.DataFrame({
    "RMSE valid": [v.loc[m, "rmse"] for m in order],
    "MAE valid": [v.loc[m, "mae"] for m in order],
    "R² valid": [v.loc[m, "r2"] for m in order],
    "RMSE test": [test_rows[m]["rmse"] for m in order],
    "MAE test": [test_rows[m]["mae"] for m in order],
    "R² test": [test_rows[m]["r2"] for m in order],
    "Доля <0 test": [test_rows[m]["negative_share"] for m in order],
}, index=order)
print("Таблица 8. Метрики регрессии")
print(table8.round(3).to_string())

fig, ax = plt.subplots(figsize=(8.5, 4.4))
x = np.arange(len(order)); w = 0.38
for i, (col, color) in enumerate([("RMSE valid", C1), ("RMSE test", C2)]):
    bars = ax.bar(x + (i - 0.5) * w, table8[col], w * 0.94, color=color, label=col.replace("RMSE ", ""))
    for b, val in zip(bars, table8[col]):
        ax.text(b.get_x() + b.get_width() / 2, val + 8, f"{val:.0f}", ha="center", fontsize=8.5)
ax.set_xticks(x, ["Среднее\ntrain", "Среднее\nпо часу", "Линейная", "Полиномиальная", "Случайный\nлес"])
ax.set(ylabel="RMSE, аренд в час", title="Рис. 7. RMSE базовых правил и трёх моделей")
ax.legend(title="Часть"); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "regression_rmse.png", dpi=160); plt.show()
''')
txt("5.6")

md("""
### 5.7. Проверка итоговой модели

В pandas передаётся только тестовая таблица (≈1,3 тыс. строк: дата, час, факт, прогноз).
""")
code(r'''
reg_local = bike_test_prediction.orderBy("date", "hour").toPandas()
reg_local["timestamp"] = (
    pd.to_datetime(reg_local["date"]) +
    pd.to_timedelta(reg_local["hour"], unit="h")
)

fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.scatter(reg_local["label"], reg_local["prediction"], s=12, alpha=0.35, color=C1)
limit = max(reg_local["label"].max(), reg_local["prediction"].max())
ax.plot([0, limit], [0, limit], "--", color=C2, label="Идеальный прогноз  y = ŷ")
ax.set(xlabel="Фактическое число аренд", ylabel="Прогноз",
       title="Рис. 8. Факт и прогноз на test")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "regression_scatter.png", dpi=160); plt.show()

week = reg_local[reg_local["timestamp"] < reg_local["timestamp"].min() + pd.Timedelta(days=7)]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(week["timestamp"], week["label"], label="Факт", color=C1, linewidth=1.8)
ax.plot(week["timestamp"], week["prediction"], label="Прогноз", color=C2, linewidth=1.8)
ax.set(xlabel="Дата и час", ylabel="Число аренд", title="Рис. 9. Первая неделя тестового периода")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "regression_week.png", dpi=160); plt.show()

mae_hour = (bike_test_prediction.groupBy("hour")
    .agg(F.avg(F.abs("residual")).alias("mae"), F.avg("label").alias("mean_label"))
    .orderBy("hour").toPandas())
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(mae_hour["hour"], mae_hour["mae"], color=C1, width=0.7)
ax.set_xticks(range(0, 24))
ax.set(xlabel="Час суток", ylabel="MAE, аренд", title="Рис. 10. Средняя абсолютная ошибка по часу суток (test)")
ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "regression_mae_by_hour.png", dpi=160); plt.show()
print(mae_hour.sort_values("mae", ascending=False).head(5).round(1).to_string(index=False))
''')
txt("5.7")

md("### 5.8. Важность признаков и ограничения")
code(r'''
rf_bike = bike_models["Случайный лес"]
bike_names = vector_names(rf_bike, bike_valid, "features")
bike_imp = (pd.Series(rf_bike.stages[-1].featureImportances.toArray(), index=bike_names)
            .sort_values(ascending=False))
top = bike_imp.head(12)[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.barh(top.index, top.values, color=C3, height=0.6)
ax.set(xlabel="Impurity-based importance", title="Рис. 11. Наиболее используемые признаки лесной регрессии")
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(OUT / "regression_importance.png", dpi=160); plt.show()
print(bike_imp.head(8).round(4).to_string())

# Отрицательные прогнозы линейных моделей и правило max(0, prediction), выбираемое по valid
print("\nОтрицательные прогнозы и обрезка max(0, ŷ):")
for name in ["Линейная регрессия", "Полиномиальная регрессия"]:
    base = bike_valid_predictions[name]
    clipped = base.withColumn("prediction_clip", F.greatest(F.col("prediction"), F.lit(0.0)))
    m0 = regression_metrics(base)
    m1 = regression_metrics(clipped, prediction="prediction_clip")
    print(f"  {name:26s} доля <0: valid {m0['negative_share']:.3f}, test {test_rows[name]['negative_share']:.3f} | "
          f"RMSE valid {m0['rmse']:.1f} -> с обрезкой {m1['rmse']:.1f}; MAE {m0['mae']:.1f} -> {m1['mae']:.1f}")
''')
txt("5.8")

# ---------------------------------------------------------------- 6
md("""
## 4. Кластеризация интернет-сессий

### 6.1–6.2. Признаки и подготовка вектора

Вектор: 6 логарифмированных счётчиков и длительностей + `BounceRates`, `ExitRates`; все 8 координат центрируются и масштабируются
(`withMean=True` допустимо: вектор плотный и маленький). `Revenue` и `PageValues` в расстояние не входят; `Revenue` подключается
только после выбора k для описания групп.
""")
code(r'''
CLUSTER_LOG_SOURCE = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration"
]
CLUSTER_LOG = ["cluster_log_" + c for c in CLUSTER_LOG_SOURCE]
CLUSTER_RATE = ["BounceRates", "ExitRates"]

cluster_expressions = [
    f"log1p(`{source}`) AS `{target}`"
    for source, target in zip(CLUSTER_LOG_SOURCE, CLUSTER_LOG)
]
cluster_prep = Pipeline(stages=[
    SQLTransformer(
        statement="SELECT *, " + ", ".join(cluster_expressions) + " FROM __THIS__"
    ),
    VectorAssembler(inputCols=CLUSTER_LOG + CLUSTER_RATE, outputCol="cluster_raw"),
    StandardScaler(inputCol="cluster_raw", outputCol="features",
                   withMean=True, withStd=True)
])

cluster_prep_model = cluster_prep.fit(shop_raw)
shop_vectors = cluster_prep_model.transform(shop_raw).select("row_id", "features").cache()

sample_order = F.sha2(F.concat_ws(
    ":", F.col("row_id").cast("string"), F.lit(str(SEED))
), 256)
cluster_sample = shop_vectors.orderBy(sample_order, "row_id").limit(4000).cache()
print("Векторов:", shop_vectors.count(), "| выборка для силуэта:", cluster_sample.count())
''')

md("""
### 6.3. KMeans и выбор k

KMeans обучается для k = 2…8 на полном распределённом наборе. Silhouette считается scikit-learn на одной и той же детерминированной выборке
4 000 строк (евклидово расстояние); дополнительно для сверки приведён силуэт Spark `ClusteringEvaluator` на всём наборе
(он использует квадрат евклидова расстояния, поэтому значения выше по шкале, но сравнимы между k).
""")
code(r'''
k_models = {}
k_rows = []
spark_silhouette = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                                       metricName="silhouette", distanceMeasure="squaredEuclidean")

for k in range(2, 9):
    sc.setJobGroup("P5_CLUSTER_" + str(k), f"KMeans k={k}")
    model = KMeans(k=k, seed=SEED, maxIter=60, tol=1e-4).fit(shop_vectors)
    sample_pred = model.transform(cluster_sample).select("features", "prediction")
    local = sample_pred.toPandas()
    x = np.vstack([v.toArray() for v in local["features"]])
    labels = local["prediction"].to_numpy()
    silhouette = silhouette_score(x, labels, metric="euclidean")

    full_pred = model.transform(shop_vectors)
    sizes = full_pred.groupBy("prediction").count()
    k_rows.append({
        "k": k,
        "silhouette": silhouette,
        "silhouette_spark_sq": spark_silhouette.evaluate(full_pred),
        "training_cost": model.summary.trainingCost,
        "min_cluster": sizes.agg(F.min("count")).first()[0]
    })
    k_models[k] = model

k_table = pd.DataFrame(k_rows).sort_values(["silhouette", "k"], ascending=[False, True])
print("Таблица 10. Подбор числа кластеров")
print(k_table.round(4).to_string(index=False))
BEST_K = int(k_table.iloc[0]["k"])
BEST_KMEANS = k_models[BEST_K]
print("Выбрано k =", BEST_K)

kt = k_table.sort_values("k")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(kt["k"], kt["silhouette"], marker="o", color=C1, linewidth=2)
axes[0].scatter([BEST_K], [kt.set_index("k").loc[BEST_K, "silhouette"]], s=120,
                facecolors="none", edgecolors=C2, linewidths=2, zorder=3)
axes[0].set(xlabel="k", ylabel="Silhouette (выборка 4 000)", title="Рис. 12. Silhouette при разном k")
axes[1].plot(kt["k"], kt["training_cost"], marker="o", color=GRAY, linewidth=2)
axes[1].set(xlabel="k", ylabel="trainingCost (WSS)", title="Внутрикластерная сумма квадратов")
plt.tight_layout(); plt.savefig(OUT / "clusters_silhouette.png", dpi=160); plt.show()
''')
txt("6.3")

md("""
### 6.4. Профили и двумерная проекция

Метки соединяются с исходными строками по `row_id`; профиль строится в исходных единицах. PCA обучается только для визуализации.
""")
code(r'''
cluster_assignments = (BEST_KMEANS.transform(shop_vectors)
    .select("row_id", F.col("prediction").alias("cluster")))

cluster_profiles = (shop_raw.join(cluster_assignments, "row_id")
    .groupBy("cluster")
    .agg(
        F.count("*").alias("n"),
        F.avg("Administrative").alias("administrative"),
        F.avg("Informational").alias("informational"),
        F.avg("ProductRelated").alias("product_pages"),
        F.avg("ProductRelated_Duration").alias("product_duration"),
        F.avg("BounceRates").alias("bounce_rate"),
        F.avg("ExitRates").alias("exit_rate"),
        F.avg(F.col("Revenue").cast("double")).alias("purchase_share")
    )
    .orderBy("cluster"))
profiles_pd = cluster_profiles.toPandas()
print("Таблица 11. Профили кластеров")
print(profiles_pd.round(3).to_string(index=False))

pca_model = PCA(k=2, inputCol="features", outputCol="pca_features").fit(shop_vectors)
print("Доля объяснённой дисперсии PC1, PC2:", np.round(pca_model.explainedVariance.toArray(), 3))
pca_rows = (pca_model.transform(BEST_KMEANS.transform(shop_vectors))
    .orderBy(sample_order, "row_id")
    .limit(4000)
    .select("prediction", "pca_features")
    .toPandas())
coords = np.vstack([v.toArray() for v in pca_rows["pca_features"]])

fig, ax = plt.subplots(figsize=(7.2, 5.2))
for cluster in sorted(pca_rows["prediction"].unique()):
    mask = pca_rows["prediction"].to_numpy() == cluster
    ax.scatter(coords[mask, 0], coords[mask, 1], s=9, alpha=0.4,
               color=PALETTE[int(cluster)], label=f"Кластер {cluster}")
ax.set(xlabel="Первая главная компонента", ylabel="Вторая главная компонента",
       title="Рис. 13. Проекция сессий на две главные компоненты")
ax.legend(markerscale=2)
plt.tight_layout(); plt.savefig(OUT / "clusters_pca.png", dpi=160); plt.show()

# Рис. 14. Нормированные профили: центры KMeans в стандартизованном пространстве
centers = pd.DataFrame(np.vstack(BEST_KMEANS.clusterCenters()),
                       columns=[c.replace("cluster_log_", "log ") for c in CLUSTER_LOG] + CLUSTER_RATE)
lim = np.abs(centers.values).max()
fig, ax = plt.subplots(figsize=(10, 0.9 + 0.75 * BEST_K))
im = ax.imshow(centers.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for i in range(centers.shape[0]):
    for j in range(centers.shape[1]):
        val = centers.values[i, j]
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(val) > lim * 0.6 else "#0b0b0b")
ax.set_xticks(range(centers.shape[1]), centers.columns, rotation=30, ha="right")
ax.set_yticks(range(centers.shape[0]), [f"Кластер {i}" for i in range(centers.shape[0])])
ax.grid(False)
ax.set_title("Рис. 14. Нормированные профили кластеров (центры, z-оценки)")
fig.colorbar(im, ax=ax, label="Отклонение от среднего, σ")
plt.tight_layout(); plt.savefig(OUT / "clusters_profiles.png", dpi=160); plt.show()
''')
txt("6.4")

# ---------------------------------------------------------------- 7
md("""
## 5. Сохранение и повторное применение моделей

Сохраняются полные `PipelineModel` (предобработка + модель), а порог классификации и параметры применения — в отдельном JSON-паспорте.
Дополнительно модели загружаются обратно и проверяется, что они воспроизводят те же предсказания.
""")
code(r'''
BEST_SHOP.write().overwrite().save((OUT / "shop_classifier").as_posix())
BEST_BIKE.write().overwrite().save((OUT / "bike_regressor").as_posix())

cluster_full_pipeline = PipelineModel(
    stages=cluster_prep_model.stages + [BEST_KMEANS]
)
cluster_full_pipeline.write().overwrite().save((OUT / "shop_kmeans").as_posix())

config = {
    "seed": SEED,
    "classification_model": BEST_SHOP_NAME,
    "classification_threshold": SHOP_THRESHOLD,
    "classification_features_note": "PageValues включён: прогноз по накопленному состоянию сессии",
    "regression_model": BEST_BIKE_NAME,
    "regression_scope": "только часы Functioning Day = Yes; погода — фактическая за час",
    "kmeans_k": BEST_K,
    "spark_version": spark.version
}
(OUT / "model_config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(config, ensure_ascii=False, indent=2))

# Проверка повторного применения.
# PipelineModel.load() в PySpark 3.5 читает metadata через Python RDD (sc.textFile(...).first()),
# а на этом стенде (Windows, Python 3.12) Python-воркер Spark аварийно завершается.
# Все стадии сохранённых цепочек — JVM-объекты, поэтому применяем Java-ридер: результат тот же PipelineModel.
from pyspark.ml.util import JavaMLReader

def load_pipeline(path):
    return JavaMLReader(PipelineModel).load(Path(path).as_posix())

cfg = json.loads((OUT / "model_config.json").read_text(encoding="utf-8"))
shop_loaded = load_pipeline(OUT / "shop_classifier")
bike_loaded = load_pipeline(OUT / "bike_regressor")
kmeans_loaded = load_pipeline(OUT / "shop_kmeans")

reloaded = binary_scores(shop_loaded, shop_test).withColumnRenamed("score", "score_loaded")
diff = shop_test_scores.join(reloaded, ["row_id", "label"]).agg(F.max(F.abs(F.col("score") - F.col("score_loaded")))).first()[0]
print("\nМакс. расхождение score классификатора после load:", diff)
print("Метрики test с порогом из JSON совпадают:",
      binary_metrics(binary_scores(shop_loaded, shop_test), cfg["classification_threshold"], ranking=False)["f1"]
      == shop_test_metrics["f1"])
print("RMSE test регрессора после load:", round(regression_metrics(bike_loaded.transform(bike_test))["rmse"], 3))
same_clusters = (kmeans_loaded.transform(shop_raw).select("row_id", "prediction")
                 .join(cluster_assignments, "row_id")
                 .filter(F.col("prediction") != F.col("cluster")).count())
print("Строк с другим кластером после load:", same_clusters)
''')
md("### Практический блок 7.1. Освобождение кэша")
code(r'''
for pred in list(shop_valid_predictions.values()) + list(bike_valid_predictions.values()):
    pred.unpersist()
for df in [shop_split, shop_test_scores, without_scores, bike_train, bike_valid,
           bike_test, bike_test_prediction, shop_vectors, cluster_sample]:
    df.unpersist()
sc.setJobGroup("P5_SELF", "Самостоятельная работа")
print("Основная часть готова. Результаты сохранены в", OUT)
print(sorted(p.name for p in OUT.iterdir()))
''')

# ---------------------------------------------------------------- 8 Самостоятельная
txt("8.intro")
md("""
### 8.1. Загрузка данных и Spark-схема

Архив распаковывается блокнотом в рабочий каталог, к строкам добавляется устойчивый `row_id` (номер строки исходного CSV).
Пустые ячейки CSV становятся `null`. Имена столбцов заменяются на короткие безопасные имена по порядку.
""")
code(r'''
POD_ZIP = ROOT / "Podcast Dataset.zip"
POD_DIR = WORK / "podcast"
with zipfile.ZipFile(POD_ZIP) as zf:
    print("Состав архива:", [(i.filename, i.file_size) for i in zf.infolist()])
    zf.extractall(POD_DIR)
POD_SOURCE = POD_DIR / "podcast_dataset.csv"
POD_WORK = WORK / "podcast_with_id.csv"

with POD_SOURCE.open("r", encoding="utf-8-sig", newline="") as src, \
     POD_WORK.open("w", encoding="utf-8", newline="") as dst:
    reader, writer = csv.reader(src), csv.writer(dst)
    pod_header = next(reader)
    writer.writerow(["row_id"] + pod_header)
    for row_id, row in enumerate(reader, start=1):
        writer.writerow([row_id] + row)

pod_schema = StructType([
    StructField("row_id", LongType(), False),
    StructField("podcast", StringType(), False),       # Podcast_Name
    StructField("episode", StringType(), False),       # Episode_Title
    StructField("length", DoubleType(), True),         # Episode_Length_minutes
    StructField("genre", StringType(), False),         # Genre
    StructField("host_pop", DoubleType(), False),      # Host_Popularity_percentage
    StructField("day", StringType(), False),           # Publication_Day
    StructField("time_of_day", StringType(), False),   # Publication_Time
    StructField("guest_pop", DoubleType(), True),      # Guest_Popularity_percentage
    StructField("ads", IntegerType(), False),          # Number_of_Ads
    StructField("sentiment", StringType(), False),     # Episode_Sentiment
    StructField("label", DoubleType(), True),          # Listening_Time_minutes (цель)
])
for original, safe in zip(["row_id"] + pod_header, pod_schema.fieldNames()):
    print(f"{original:30s} -> {safe}")

pod_raw = (spark.read
    .option("header", True)
    .option("mode", "FAILFAST")
    .schema(pod_schema)
    .csv(POD_WORK.resolve().as_posix())
    .cache())
print("\nСтрок:", pod_raw.count())
pod_raw.printSchema()
pod_raw.show(5, truncate=False)
''')

md("""
### 8.2. Распределение ключей объектов и схема разбиения

Перед разбиением нужно понять, что является «объектом»: строка, эпизод (пара подкаст + название) или подкаст целиком.
Если один объект встречается в нескольких строках, случайное построчное разбиение отправит его копии и в train, и в test — это утечка.
""")
code(r'''
n_rows = pod_raw.count()
key_stats = pod_raw.agg(
    F.countDistinct("podcast").alias("podcasts"),
    F.countDistinct("episode").alias("episode_titles"),
    F.countDistinct("podcast", "episode").alias("podcast_episode_pairs"),
).first().asDict()
print("Строк:", n_rows, "|", key_stats)

per_pair = pod_raw.groupBy("podcast", "episode").agg(
    F.count("*").alias("rows"),
    F.countDistinct("length").alias("distinct_length"),
    F.countDistinct("genre").alias("distinct_genre"),
    F.countDistinct("day").alias("distinct_day"))
per_pair.select(
    F.min("rows"), F.expr("percentile(rows, 0.5)").alias("median_rows"), F.max("rows"),
    F.avg("distinct_length").alias("avg_distinct_length"),
    F.max("distinct_genre").alias("max_genre_per_pair"),
    F.avg("distinct_day").alias("avg_distinct_day")).show()

per_podcast = pod_raw.groupBy("podcast").agg(F.count("*").alias("rows"),
                                             F.countDistinct("genre").alias("genres"))
per_podcast.select(F.min("rows"), F.max("rows"), F.max("genres").alias("max_genres_per_podcast")).show()

value_cols = [c for c in pod_raw.columns if c != "row_id"]
dup_groups = pod_raw.groupBy(*value_cols).count().filter("count > 1")
print("Полностью совпадающих строк (кроме row_id):",
      dup_groups.agg(F.sum("count")).first()[0], "в", dup_groups.count(), "группах;",
      "лишних копий:", n_rows - pod_raw.select(*value_cols).distinct().count())
print("Пропуски:", {k: v for k, v in null_counts(pod_raw).items() if v})

pp = per_pair.select("rows").toPandas()     # 4 799 строк — безопасно
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.hist(pp["rows"], bins=range(1, pp["rows"].max() + 2), color=C1, edgecolor="white")
ax.set(xlabel="Строк на пару (подкаст, название эпизода)", ylabel="Число пар",
       title="Распределение числа строк на ключ эпизода")
ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "self_key_distribution.png", dpi=160); plt.show()
''')
txt("8.2")

md("""
### 8.3. Разбиение на train / valid / test

1. Удаляются полные дубли (остаётся строка с минимальным `row_id`).
2. Отделяются строки без целевой переменной — на них нельзя ни обучать, ни оценивать (в эксплуатации это как раз «новые» эпизоды, для которых нужен прогноз).
3. Разбиение 60/20/20 выполняется **по группам** — паре (подкаст, эпизод) — через SHA-256 от ключа и seed. Все строки одной пары попадают в одну часть;
   результат не зависит от порядка партиций и воспроизводится при повторном запуске.
""")
code(r'''
pod_dedup = pod_raw.groupBy(*value_cols).agg(F.min("row_id").alias("row_id"))
pod_unlabelled = pod_dedup.filter(F.col("label").isNull())
pod_labelled = pod_dedup.filter(F.col("label").isNotNull())

group_hash = F.sha2(F.concat_ws(":", "podcast", "episode", F.lit(str(SEED))), 256)
bucket = F.pmod(F.conv(F.substring(group_hash, 1, 8), 16, 10).cast("long"), F.lit(100))
pod_split = (pod_labelled
    .withColumn("bucket", bucket)
    .withColumn("split", F.when(F.col("bucket") < 60, "train")
                          .when(F.col("bucket") < 80, "valid")
                          .otherwise("test"))
    .drop("bucket")
    .cache())

print("После удаления дублей:", pod_dedup.count(), "| без цели:", pod_unlabelled.count(),
      "| с целью:", pod_split.count())
(pod_split.groupBy("split")
    .agg(F.count("*").alias("rows"),
         F.countDistinct("podcast", "episode").alias("pairs"),
         F.countDistinct("podcast").alias("podcasts"),
         F.round(F.avg("label"), 2).alias("mean_label"),
         F.round(F.stddev("label"), 2).alias("std_label"),
         F.round(F.avg(F.col("length").isNull().cast("double")), 3).alias("length_null_share"))
    .orderBy(F.expr("CASE split WHEN 'train' THEN 0 WHEN 'valid' THEN 1 ELSE 2 END"))
    .show())

crossing = (pod_split.groupBy("podcast", "episode").agg(F.countDistinct("split").alias("parts"))
            .filter("parts > 1").count())
print("Пар, попавших в несколько частей:", crossing)
assert crossing == 0

pod_train = pod_split.filter("split = 'train'").drop("split").cache()
pod_valid = pod_split.filter("split = 'valid'").drop("split").cache()
pod_test = pod_split.filter("split = 'test'").drop("split").cache()
print("train / valid / test:", pod_train.count(), pod_valid.count(), pod_test.count())
''')
txt("8.3")

md("""
### 8.4. Распределения предикторов и целевой переменной

Разведочный анализ выполняется **только на train**, чтобы наблюдения valid/test не влияли на выбор признаков. Агрегаты считаются в Spark,
в pandas переносятся только числовые столбцы train (≈28 тыс. строк × 6 столбцов) для гистограмм.
""")
code(r'''
eda = (pod_train
       .select("label", "length", "host_pop", "guest_pop", "ads",
               (F.col("label") / F.col("length")).alias("listen_share"))
       .toPandas())
print(eda.describe().T.round(2).to_string())

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
spec = [("label", "Время прослушивания, мин (цель)", C2), ("length", "Длительность эпизода, мин", C1),
        ("host_pop", "Популярность ведущего, %", C1), ("guest_pop", "Популярность гостя, %", C1),
        ("listen_share", "Доля прослушанного = цель / длительность", GRAY)]
for ax, (col, title, color) in zip(axes.flat, spec):
    ax.hist(eda[col].dropna(), bins=40, color=color, edgecolor="white")
    ax.set_title(title, fontsize=10, fontweight="normal"); ax.grid(axis="x", visible=False)
ads = eda["ads"].value_counts().sort_index()
axes.flat[5].bar(ads.index.astype(str), ads.values, color=C1, width=0.6)
axes.flat[5].set_title("Число рекламных вставок", fontsize=10, fontweight="normal")
axes.flat[5].grid(axis="x", visible=False)
fig.suptitle("Распределения предикторов и цели (train)", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "self_distributions.png", dpi=160); plt.show()
''')
code(r'''
print("Корреляция Пирсона с целью (train, попарно по непустым):")
print(eda.corr()["label"].drop("label").round(3).to_string())

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, col, title in zip(axes, ["genre", "day", "time_of_day", "sentiment"],
                          ["Жанр", "День публикации", "Время публикации", "Тональность"]):
    g = (pod_train.groupBy(col).agg(F.avg("label").alias("mean"), F.count("*").alias("n"))
         .orderBy("mean").toPandas())
    ax.barh(g[col], g["mean"], color=C1, height=0.6)
    ax.axvline(eda["label"].mean(), color=GRAY, linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=10); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Среднее время, мин")
    ax.set_xlim(g["mean"].min() - 3, g["mean"].max() + 2)
    ax.tick_params(axis="y", labelleft=True, labelsize=8)
fig.suptitle("Среднее время прослушивания по категориям (train; пунктир — общее среднее)", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "self_categories.png", dpi=160); plt.show()

sample = eda.sample(4000, random_state=SEED)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(sample["length"], sample["label"], s=8, alpha=0.3, color=C1)
ax.plot([0, 120], [0, 120], "--", color=C2, label="цель = длительность")
ax.set(xlabel="Длительность эпизода, мин", ylabel="Время прослушивания, мин",
       title="Цель против длительности (train, 4 000 точек)")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "self_length_scatter.png", dpi=160); plt.show()

miss = (pod_train.agg(*[F.avg(F.col(c).isNull().cast("double")).alias(c) for c in ["length", "guest_pop"]])
        .first().asDict())
print("Доля пропусков в train:", {k: round(v, 3) for k, v in miss.items()})
print("Среднее цели при пропуске / наличии length:",
      [round(r["m"], 2) for r in pod_train.groupBy(F.col("length").isNull().alias("na"))
       .agg(F.avg("label").alias("m")).orderBy("na").collect()])
print("Среднее цели при пропуске / наличии guest_pop:",
      [round(r["m"], 2) for r in pod_train.groupBy(F.col("guest_pop").isNull().alias("na"))
       .agg(F.avg("label").alias("m")).orderBy("na").collect()])
''')
txt("8.4")

txt("8.5")
code(r'''
POD_CAT = ["podcast", "genre", "day", "time_of_day", "sentiment"]
POD_NUM = ["length_imp", "host_pop", "guest_imp", "ads", "length_missing", "guest_missing"]

def pod_pipeline(kind, **params):
    flags = SQLTransformer(statement=(
        "SELECT *, CAST(length IS NULL AS DOUBLE) AS length_missing, "
        "CAST(guest_pop IS NULL AS DOUBLE) AS guest_missing FROM __THIS__"))
    imputer = Imputer(inputCols=["length", "guest_pop"], outputCols=["length_imp", "guest_imp"],
                      strategy="median")
    index_cols = [c + "_idx" for c in POD_CAT]
    encoded_cols = [c + "_ohe" for c in POD_CAT]
    indexer = StringIndexer(inputCols=POD_CAT, outputCols=index_cols,
                            handleInvalid="keep", stringOrderType="alphabetAsc")
    encoder = OneHotEncoder(inputCols=index_cols, outputCols=encoded_cols,
                            handleInvalid="keep", dropLast=False)
    stages = [flags, imputer, indexer, encoder]
    if kind == "LINEAR":
        stages += [
            VectorAssembler(inputCols=POD_NUM + encoded_cols, outputCol="features_raw"),
            StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True),
            LinearRegression(maxIter=200, standardization=False,
                             regParam=params.get("regParam", 0.0),
                             elasticNetParam=params.get("elasticNetParam", 0.0)),
        ]
    elif kind == "RF":
        stages += [
            VectorAssembler(inputCols=POD_NUM + encoded_cols, outputCol="features"),
            RandomForestRegressor(seed=SEED, maxBins=64, featureSubsetStrategy="onethird",
                                  numTrees=params.get("numTrees", 60), maxDepth=params.get("maxDepth", 8),
                                  minInstancesPerNode=params.get("minInstancesPerNode", 5)),
        ]
    elif kind == "GBT":
        stages += [
            VectorAssembler(inputCols=POD_NUM + encoded_cols, outputCol="features"),
            GBTRegressor(seed=SEED, maxBins=64, stepSize=params.get("stepSize", 0.1),
                         maxIter=params.get("maxIter", 50), maxDepth=params.get("maxDepth", 5),
                         subsamplingRate=0.8),
        ]
    else:
        raise ValueError(kind)
    return Pipeline(stages=stages)

# Базовые правила (статистики только по train)
pod_train_mean = pod_train.agg(F.avg("label")).first()[0]
share_stats = pod_train.filter(F.col("length").isNotNull()).agg(
    F.avg(F.col("label") / F.col("length")).alias("share")).first()
POD_SHARE = share_stats["share"]
pod_mean_no_length = pod_train.filter(F.col("length").isNull()).agg(F.avg("label")).first()[0]

def pod_baseline(df, rule):
    if rule == "mean":
        return df.withColumn("prediction", F.lit(pod_train_mean))
    if rule == "share":   # средняя доля прослушивания × длительность эпизода
        return df.withColumn("prediction", F.when(F.col("length").isNotNull(),
                             F.col("length") * F.lit(POD_SHARE)).otherwise(F.lit(pod_mean_no_length)))
    raise ValueError(rule)

print(f"Среднее train = {pod_train_mean:.2f} мин; средняя доля прослушивания = {POD_SHARE:.3f}; "
      f"среднее при пропущенной длительности = {pod_mean_no_length:.2f} мин")
''')

md("""
### 8.6. Обучение моделей и подбор гиперпараметров

Подбор выполняется **минимизацией RMSE на valid** по небольшим сеткам для трёх семейств: линейная регрессия (сила и тип регуляризации),
случайный лес (число и глубина деревьев), градиентный бустинг GBT (глубина и число итераций). Для линейной модели дополнительно
выполнена 3-кратная кросс-валидация на train (`CrossValidator`) — чтобы сравнить два способа подбора.
""")
code(r'''
grids = {
    "LINEAR": [dict(regParam=r, elasticNetParam=e) for r in [0.0, 0.01, 0.1, 1.0] for e in [0.0, 1.0]
               if not (r == 0.0 and e == 1.0)],
    "RF": [dict(numTrees=t, maxDepth=d) for t in [60, 120] for d in [6, 10, 14]],
    "GBT": [dict(maxDepth=d, maxIter=i) for d in [3, 5, 7] for i in [40, 100]],
}
FAMILY = {"LINEAR": "Линейная регрессия", "RF": "Случайный лес", "GBT": "Градиентный бустинг (GBT)"}

tuning_rows = []
for kind, grid in grids.items():
    for params in grid:
        sc.setJobGroup("P5_SELF_" + kind, f"{FAMILY[kind]} {params}")
        t0 = perf_counter()
        model = pod_pipeline(kind, **params).fit(pod_train)
        fit_s = perf_counter() - t0
        m = regression_metrics(model.transform(pod_valid))
        tuning_rows.append({"kind": kind, "params": params, "fit_s": fit_s, **m, "_model": model})
        print(f"{FAMILY[kind]:26s} {str(params):40s} RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} R2={m['r2']:.4f} ({fit_s:.1f} c)")

tuning = pd.DataFrame(tuning_rows)
''')
code(r'''
# Лучшая конфигурация каждого семейства по RMSE valid + базовые правила
best_per_family = tuning.loc[tuning.groupby("kind")["rmse"].idxmin()].set_index("kind")
pod_models = {FAMILY[k]: best_per_family.loc[k, "_model"] for k in best_per_family.index}

summary_rows = []
for k in ["LINEAR", "RF", "GBT"]:
    row = best_per_family.loc[k]
    train_m = regression_metrics(row["_model"].transform(pod_train))
    summary_rows.append({"Решение": FAMILY[k], "Гиперпараметры": str(row["params"]),
                         "RMSE train": train_m["rmse"], "RMSE valid": row["rmse"],
                         "MAE valid": row["mae"], "R² valid": row["r2"], "fit, c": row["fit_s"]})
for rule, title in [("mean", "Базовое: среднее train"), ("share", "Базовое: доля × длительность")]:
    m = regression_metrics(pod_baseline(pod_valid, rule))
    summary_rows.append({"Решение": title, "Гиперпараметры": "—", "RMSE train":
                         regression_metrics(pod_baseline(pod_train, rule))["rmse"],
                         "RMSE valid": m["rmse"], "MAE valid": m["mae"], "R² valid": m["r2"], "fit, c": 0.0})
pod_validation = pd.DataFrame(summary_rows).sort_values("RMSE valid")
print(pod_validation.round(3).to_string(index=False))

learned = pod_validation[pod_validation["Решение"].isin(pod_models)]
BEST_POD_NAME = str(learned.iloc[0]["Решение"])
BEST_POD = pod_models[BEST_POD_NAME]
print("\nВыбрана модель:", BEST_POD_NAME, "|", learned.iloc[0]["Гиперпараметры"])

# Кривые подбора
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
lin = tuning[tuning["kind"] == "LINEAR"].copy()
lin["label"] = lin["params"].apply(lambda p: f"λ={p['regParam']}\n{'L1' if p['elasticNetParam'] else 'L2'}")
# точечная диаграмма: различия малы, а столбцы с ненулевой базой исказили бы пропорции
axes[0].plot(lin["label"], lin["rmse"], "o", color=C1, markersize=8)
axes[0].set(title="Линейная регрессия", ylabel="RMSE valid, мин"); axes[0].tick_params(axis="x", labelsize=8)
for ax, kind, series_key, x_key, colors in [(axes[1], "RF", "numTrees", "maxDepth", [C2, "#a23d14"]),
                                             (axes[2], "GBT", "maxIter", "maxDepth", [C3, "#0d6b4a"])]:
    part = tuning[tuning["kind"] == kind]
    for (val, grp), color in zip(part.groupby(part["params"].apply(lambda p: p[series_key])), colors):
        xs = grp["params"].apply(lambda p: p[x_key])
        ax.plot(xs, grp["rmse"], marker="o", color=color, linewidth=2, label=f"{series_key}={val}")
    ax.set(title=FAMILY[kind], xlabel=x_key, ylabel="RMSE valid, мин"); ax.legend()
fig.suptitle("Подбор гиперпараметров по RMSE на valid", fontweight="bold")
plt.tight_layout(); plt.savefig(OUT / "self_tuning.png", dpi=160); plt.show()
''')
code(r'''
# Кросс-валидация для линейной модели (3 фолда на train): все преобразования переобучаются внутри фолдов
lin_pipe = pod_pipeline("LINEAR")
lin_est = lin_pipe.getStages()[-1]
cv_grid = (ParamGridBuilder()
           .addGrid(lin_est.regParam, [0.0, 0.01, 0.1, 1.0])
           .addGrid(lin_est.elasticNetParam, [0.0, 1.0])
           .build())
cv = CrossValidator(estimator=lin_pipe, estimatorParamMaps=cv_grid,
                    evaluator=RegressionEvaluator(labelCol="label", metricName="rmse"),
                    numFolds=3, seed=SEED, parallelism=2)
sc.setJobGroup("P5_SELF_CV", "CrossValidator для линейной регрессии")
t0 = perf_counter()
cv_model = cv.fit(pod_train)
print(f"CrossValidator: {len(cv_grid)} конфигураций × 3 фолда за {perf_counter() - t0:.1f} c")
cv_table = pd.DataFrame([{"regParam": pm[lin_est.regParam], "elasticNet": pm[lin_est.elasticNetParam], "RMSE CV": r}
                         for pm, r in zip(cv_grid, cv_model.avgMetrics)]).sort_values("RMSE CV")
print(cv_table.round(4).to_string(index=False))
best_cv = cv_model.bestModel.stages[-1]
print("Лучшее по CV: regParam =", best_cv.getRegParam(), ", elasticNet =", best_cv.getElasticNetParam(),
      "| RMSE valid =", round(regression_metrics(cv_model.bestModel.transform(pod_valid))["rmse"], 3))
''')
txt("8.6")

md("""
### 8.7. Однократная оценка на test и визуализация

Выбранная по valid модель применяется к test один раз; базовые правила — фиксированные формулы со статистиками train.
Остальные модели пересчитаны на test только справочно, после фиксации выбора.
""")
code(r'''
sc.setJobGroup("P5_SELF_TEST", "Самостоятельная: test")
pod_test_pred = (BEST_POD.transform(pod_test)
    .select("row_id", "podcast", "genre", "length", "label", "prediction")
    .withColumn("residual", F.col("label") - F.col("prediction"))
    .cache())
pod_test_metrics = regression_metrics(pod_test_pred)
print("TEST", BEST_POD_NAME, {k: round(v, 4) for k, v in pod_test_metrics.items()})

final_rows = []
for name, model in pod_models.items():
    m = pod_test_metrics if name == BEST_POD_NAME else regression_metrics(model.transform(pod_test))
    final_rows.append({"Решение": name + (" (выбрана)" if name == BEST_POD_NAME else " (справочно)"), **m})
for rule, title in [("mean", "Базовое: среднее train"), ("share", "Базовое: доля × длительность")]:
    final_rows.append({"Решение": title, **regression_metrics(pod_baseline(pod_test, rule))})
pod_final = pd.DataFrame(final_rows).sort_values("rmse")
print("\nМетрики на test:")
print(pod_final.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8.5, 4))
colors = [C1 if "выбрана" in n else (GRAY if "Базовое" in n else "#9cc0ec") for n in pod_final["Решение"]]
bars = ax.barh(pod_final["Решение"][::-1], pod_final["rmse"][::-1], color=colors[::-1], height=0.6)
for b, val in zip(bars, pod_final["rmse"][::-1]):
    ax.text(val + 0.2, b.get_y() + b.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
ax.set(xlabel="RMSE на test, мин", title="Сравнение с базовыми правилами на test")
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(OUT / "self_test_rmse.png", dpi=160); plt.show()
''')
code(r'''
loc = pod_test_pred.toPandas()     # test ≈ 9 тыс. строк × 6 столбцов
s = loc.sample(4000, random_state=SEED)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].scatter(s["label"], s["prediction"], s=8, alpha=0.3, color=C1)
axes[0].plot([0, 120], [0, 120], "--", color=C2, label="y = ŷ")
axes[0].set(xlabel="Фактическое время, мин", ylabel="Прогноз, мин", title=f"Факт и прогноз на test ({BEST_POD_NAME})")
axes[0].legend()
axes[1].hist(loc["residual"], bins=60, color=C1, edgecolor="white")
axes[1].axvline(0, color=C2, linestyle="--")
axes[1].set(xlabel="Остаток = факт − прогноз, мин", ylabel="Число эпизодов", title="Распределение остатков на test")
axes[1].grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "self_test_scatter.png", dpi=160); plt.show()

# Ошибка по группам длительности (включая пропуск длительности)
by_len = (pod_test_pred
    .withColumn("length_group", F.when(F.col("length").isNull(), "нет данных")
        .when(F.col("length") < 30, "<30").when(F.col("length") < 60, "30–60")
        .when(F.col("length") < 90, "60–90").otherwise("≥90"))
    .groupBy("length_group").agg(F.count("*").alias("n"), F.avg(F.abs("residual")).alias("mae"),
                                  F.avg("residual").alias("bias"))
    .toPandas().set_index("length_group").loc[["<30", "30–60", "60–90", "≥90", "нет данных"]])
print(by_len.round(2).to_string())
fig, ax = plt.subplots(figsize=(7.5, 3.8))
bars = ax.bar(by_len.index, by_len["mae"], color=[C1] * 4 + [C2], width=0.6)
for b, (val, n) in zip(bars, zip(by_len["mae"], by_len["n"])):
    ax.text(b.get_x() + b.get_width() / 2, val + 0.2, f"{val:.1f}\n(n={n})", ha="center", fontsize=8.5)
ax.set(xlabel="Длительность эпизода, мин", ylabel="MAE на test, мин",
       title="Ошибка в зависимости от длительности эпизода", ylim=(0, by_len["mae"].max() * 1.25))
ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.savefig(OUT / "self_mae_by_length.png", dpi=160); plt.show()
''')
code(r'''
# Важность признаков выбранной модели (для деревьев) или коэффициенты (для линейной)
last = BEST_POD.stages[-1]
if hasattr(last, "featureImportances"):
    names = vector_names(BEST_POD, pod_valid, "features")
    imp = pd.Series(last.featureImportances.toArray(), index=names).sort_values(ascending=False)
    xlabel = "Impurity-based importance"
else:
    names = vector_names(BEST_POD, pod_valid, "features_raw")
    imp = pd.Series(np.abs(last.coefficients.toArray()), index=names).sort_values(ascending=False)
    xlabel = "|коэффициент| (масштабированные признаки)"
top = imp.head(12)[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(top.index, top.values, color=C1, height=0.6)
ax.set(xlabel=xlabel, title=f"Наиболее важные признаки: {BEST_POD_NAME}")
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(OUT / "self_importance.png", dpi=160); plt.show()
print(imp.head(8).round(4).to_string())

# Сохранение модели и паспорта применения
BEST_POD.write().overwrite().save((OUT / "podcast_regressor").as_posix())
pod_config = {
    "seed": SEED, "task": "регрессия Listening_Time_minutes",
    "model": BEST_POD_NAME, "hyperparameters": str(learned.iloc[0]["Гиперпараметры"]),
    "split": "группы (podcast, episode), SHA-256 + seed, 60/20/20, после удаления полных дублей",
    "baseline_share": POD_SHARE, "test_rmse": pod_test_metrics["rmse"], "test_mae": pod_test_metrics["mae"],
    "spark_version": spark.version,
}
(OUT / "podcast_config.json").write_text(json.dumps(pod_config, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(pod_config, ensure_ascii=False, indent=2))
''')
txt("8.7")

# ---------------------------------------------------------------- 9, 10
txt("9")
txt("10")
code(r'''
for df in [pod_raw, pod_split, pod_train, pod_valid, pod_test, pod_test_pred]:
    df.unpersist()
spark.stop()
print("SparkSession остановлена.")
''')

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
out = Path(__file__).resolve().parent.parent / "Практика_5.ipynb"
nbf.write(nb, out)
print("written", out, len(C), "cells")
