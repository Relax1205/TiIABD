"""Сборка самостоятельного, выполняемого сверху вниз блокнота ПР6."""
from pathlib import Path
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
B = json.loads((ROOT / '_build/blocks.json').read_text(encoding='utf-8'))
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))
def block(key, text):
    md(text)
    code(B[key])

md('''# Практическая работа №6. Обработка текстовых данных с использованием Spark MLlib

| | |
|---|---|
| Дисциплина | Технологии и инструментарий анализа больших данных |
| Студент | Дицель Н.С. |
| Группа | ИКБО-11-23 |
| Проверил | старший преподаватель Приходько Н.А. |
| Дата | 02.11.2026 |

**Цель:** подготовить текстовые признаки, сравнить классификаторы тональности и авторства, построить генераторы Маркова первого и второго порядка.

Содержание: окружение и контроль данных → подготовка и EDA → BoW/TF–IDF → тональность → сохранение → Марков → самостоятельная работа Literature → выводы.

Блокнот выполняется сверху вниз. Архивы данных остаются рядом с ним; распаковка и промежуточные файлы создаются автоматически. Обработка корпусов, обучение и агрегации выполняются в Spark, локально — небольшие итоговые таблицы, графики и последовательная генерация. Контрольные числа из методички не подставляются вместо фактических расчётов.''')
md('## 1. Окружение и чтение корпуса (раздел 3 методички)')
code(r'''
from pathlib import Path
from time import perf_counter
from collections import Counter, defaultdict
from zipfile import ZipFile
import csv, hashlib, json, math, random, re, sys, os, platform, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "sentiment_dataset.zip").exists():
    ROOT = ROOT / "Практическая работа №6"
assert (ROOT / "Literature.zip").exists(), ROOT
WORK, OUT = ROOT / "practice6_data", ROOT / "practice6_results"
WORK.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if os.name == "nt":
    hadoop = Path(r"C:\hadoop")
    if not (hadoop / "bin/hadoop.dll").exists():
        hadoop = ROOT.parent / "Практическая работа №3/hadoop"
    if (hadoop / "bin/hadoop.dll").exists():
        os.environ["HADOOP_HOME"] = str(hadoop)
        os.environ["PATH"] = str(hadoop / "bin") + os.pathsep + os.environ["PATH"]
        _dll = os.add_dll_directory(str(hadoop / "bin"))

import pyspark
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (SQLTransformer, RegexTokenizer, StopWordsRemover,
    CountVectorizer, CountVectorizerModel, IDF, Normalizer, NGram, HashingTF)
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.functions import vector_to_array
from pyspark.ml.util import JavaMLReader

def load_pipeline(path):
    # Все стадии — JVM-объекты. Java-ридер обходит Python RDD,
    # используемый стандартным PipelineModel.load в PySpark 3.5.
    return JavaMLReader(PipelineModel).load(Path(path).as_posix())
SEED = 42
spark = (SparkSession.builder.master("local[2]").appName("Practice6_TextAnalytics")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.showConsoleProgress", "false").getOrCreate())
sc = spark.sparkContext
sc.setLogLevel("ERROR")
LABEL_NAMES = {0: "Нейтральный", 1: "Положительный", 2: "Отрицательный"}
plt.rcParams.update({"figure.dpi": 100, "font.family": "DejaVu Sans"})
print("Каталог:", ROOT)
print("Python:", platform.python_version(), "Spark:", spark.version, "master:", sc.master)
print(subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0])
SOURCE = WORK / "sentiment_dataset.csv"
if not SOURCE.exists():
    with ZipFile(ROOT / "sentiment_dataset.zip") as archive:
        with archive.open("sentiment_dataset.csv") as src, SOURCE.open("wb") as dst:
            import shutil
            shutil.copyfileobj(src, dst)

# Малые локальные примеры передаются через JSONL: это не требует
# Python worker на Windows. Корпуса и обучение остаются в Spark.
def local_frame(rows, columns):
    import uuid
    path = WORK / ("small_" + uuid.uuid4().hex + ".jsonl")
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(zip(columns, row)), ensure_ascii=False) + "\n")
    return spark.read.json(path.as_posix())
''')
block('3.2', '### 1.1. Потоковое чтение многострочного CSV\n\nСначала проверяем формат CSV и отбираем rureviews. Затем Spark читает JSONL с явной схемой.')
B['3.3'] = B['3.3'].replace('assert bad_text == 0, "Пустые тексты требуют отдельного решения"',
    'print("Пустые/пробельные тексты сохраняются: это проверка обработки нулевых признаков, а не основание менять test.")')
block('3.3', '### 1.2. Схема, пустые тексты, дубли и Parquet\n\nМетки: 0 — нейтрально, 1 — положительно, 2 — отрицательно. По 28 812 записей каждого класса. В исходном корпусе обнаружен один пробельный отзыв; сохраняем его и учитываем среди нулевых векторов. Это уточняет контрольные значения методички.')
block('4.1', '## 2. Подготовка и независимое разбиение (раздел 4)\n\nНижний регистр, сохранённые числа, дефисы, отрицания и эмоциональная пунктуация. Приведённый в методичке шаблон не включает эмодзи: сохраняем его для воспроизведения разбиения; это ограничение основной токенизации.')
block('4.2', '### 2.1. Группы одинаковых последовательностей токенов\n\nSHA-256 группы без метки → фиксированное разбиение 70/15/15. Противоречивые метки сохраняются, повторы остаются в одной части. CountVectorizer и IDF ещё не обучались.')
block('4.3', '### 2.2. Стоп-слова\n\nСравним исходный список, SAFE_STOP и сохранение всех слов. «Не» и «очень» важны для тональности. DISPLAY_STOP применяется только к рисункам. Лемматизация в основном опыте не используется: словоформы сохраняют стиль и грамматику.')
block('5.1', '## 3. Исследование обучающих отзывов (раздел 5)\n\nРаспределение классов и длин строится только по train; в pandas переносятся агрегаты.')
md('**Вывод:** классы train близки по объёму, но длины отзывов различаются. Длинные отзывы содержат больше потенциальных признаков; TF–IDF дополнительно нормируется.')
block('5.2', '### 3.1. TF, DF и ранговый график\n\nTF — число вхождений, DF — число документов с токеном. Частое повторение в одном отзыве увеличивает только TF.')
md('**Вывод:** небольшой набор частых слов даёт большую долю вхождений, тогда как редкие слова образуют длинный хвост. Для модели minDF отсекает самые редкие признаки.')
block('5.3', '### 3.2. Слова по классам и облако слов\n\nДоли нормируются числом документов каждого класса, поэтому сравнение не зависит от размера класса. Облако слов — дополнительная иллюстрация.')
md('**Вывод:** «спасибо» и «отличное» связаны с положительными оценками; отрицание встречается в разных классах и требует контекста. Размер слова в облаке не показывает его причинного влияния.')
block('6.1', '## 4. Ручная проверка признаков (раздел 6)\n\nНа трёх документах сверим словарь, координаты BoW и разреженное хранение.')
code('''
toy_rows = toy_counts.orderBy("id").collect()
manual_df = np.array([sum(term in r.tokens for r in toy_rows) for term in toy_cv.vocabulary])
manual_idf = np.log((len(toy_rows) + 1) / (manual_df + 1))
print("DF вручную:", manual_df, "IDF вручную:", manual_idf)
''')
block('6.2', '### 4.1. TF–IDF и L2-нормирование\n\nSpark использует IDF = ln((N+1)/(DF+1)); TF — счётчик, без деления на длину. Слово во всех документах получает IDF=0.')
code('assert np.allclose(toy_idf.idf.toArray(), manual_idf)\nprint("Ручной IDF совпадает со Spark")')
block('6.3', '### 4.2. Биграммы и HashingTF\n\nHashingTF не обучает словарь, но возможны коллизии. CountVectorizer игнорирует неизвестные слова.')
block('7.1', '## 5. Классификация тональности (раздел 7)\n\nЧетыре полных Pipeline: NB BoW, LR BoW, LR TF–IDF и LR TF–IDF с биграммами. Словарь до 20 000, minDF=3, λ=0,001. Fit выполняется только на train. Все результаты valid используются для выбора, test — после фиксации решения.')
block('7.2', '### 5.1. Macro-F1, weighted-F1, log loss и постоянное правило\n\nMacro-F1 даёт классам равный вес; weighted-F1 учитывает их численность. Матрица ошибок считается в Spark, в NumPy переносится только 3×3.')
block('7.3', '### 5.2. Обучение и valid\n\nВремя измеряется для всей цепочки fit, включая словарь и IDF.')
B['7.4'] = B['7.4'].replace('RUN_ABLATION = False', 'RUN_ABLATION = True')
block('7.4', '### 5.3. Проверка стоп-листов\n\nДва дополнительных варианта LR TF–IDF сравниваются на том же valid. Результаты включены в общий выбор.')
block('8.1', '## 6. Итоговый test и анализ ошибок (раздел 8)\n\nСначала фиксируется максимальный macro-F1 valid, затем выбранная модель оценивается на test. Остальные модели по test не подбираются.')
code('''
test_baseline_metrics, _, _ = multiclass_metrics(test.withColumn("prediction", F.lit(majority)))
print("Постоянный класс, test:", test_baseline_metrics)
''')
B['8.2'] = B['8.2'].replace(
    'nnz = F.udf(lambda v: int(v.numNonzeros()), IntegerType())',
    'def nnz(col):\n    return F.size(F.filter(F.unwrap_udt(F.col(col)).getField("values"), lambda x: x != 0))')
block('8.2', '### 6.1. OOV, ненулевые координаты и ошибочные отзывы\n\nOOV считается по вхождениям входных токенов CountVectorizer. Нулевые векторы остаются в оценке качества. nnz вычисляется нативным выражением Spark по значениям разреженного вектора, без Python UDF и разворачивания полного словаря.')
code('''
sentiment_coverage = dict(oov_fraction=oov_n/max(term_n,1),
    zero_vectors=coverage.filter("nnz = 0").count(),
    mean_nnz=float(coverage.agg(F.avg("nnz")).first()[0]))
(OUT / "sentiment_coverage.json").write_text(json.dumps(sentiment_coverage, indent=2), encoding="utf-8")
''')
block('8.3', '### 6.2. Различающие слова\n\nКонтраст w(положительный) − w(отрицательный) относится к логарифму отношения вероятностей классов. Это статистическая связь, не причинный эффект.')
block('9.1', '## 7. Сохранение и применение (раздел 9)\n\nСохраняется вся PipelineModel вместе с параметрами опыта. Пустой вход явно отмечается как не имеющий признаков.')
code('''
assert all(x.prediction == y.prediction and np.allclose(x.probability, y.probability)
           for x, y in zip(a,b))
print("После загрузки совпадают классы и вероятности всех четырёх примеров.")
''')
block('10.1', '## 8. Генерация отзывов цепями Маркова (раздел 10)\n\nИспользуются 3 000 фиксированных train-отзывов. Предложения выделяются внутри документов; словоформы, стоп-слова и пунктуация сохраняются. Разделение по пунктуации приближённое: сокращения могут разделяться ошибочно.')
block('10.2', '### 8.1. Spark NGram и таблицы переходов\n\nПорядок 1: биграммы; порядок 2: триграммы. BOS/EOS не допускают переходов между предложениями. Локальный перенос ограничен числом переходов.')
block('10.3', '### 8.2. Сэмплирование и backoff\n\nШесть примеров с фиксированными seed. Неизвестная пара → последнее слово → старт предложения. Temperature меняет частотные веса, max_tokens ограничивает число новых токенов.')
block('10.4', '### 8.3. Вероятности продолжений\n\nДля каждого состояния P(w|h)=count(h,w)/Σcount(h,w). Проверяем воспроизводимость и ограничение длины.')
code('''
for frame in valid_predictions.values(): frame.unpersist()
test_predictions.unpersist(); eda.unpersist(); split_df.unpersist()
sentences.unpersist(); transitions.unpersist()
''')

# Самостоятельная часть добавляется из отдельного файла построения ячеек.
exec((ROOT / '_build/literature_cells.py').read_text(encoding='utf-8'))
md('## 10. Краткие ответы на вопросы\n\nПолные формулировки и краткие ответы по лекции №6 «Глубокое обучение» (слайд 64/64) и все 18 вопросов раздела 12 практики №6 приведены в `Ответы_на_вопросы_6.md`.')
md('## 11. Итоговые выводы\n\nФактические метрики, разбор ошибок тональности и авторства, сравнение генераторов и ограничения приведены в итоговом отчёте `Результаты_практики_6.md`. Метрики также сохранены в JSON/CSV. Выбор моделей выполнен по valid, тестовые произведения независимы от обучающих.')
code('spark.stop()\nprint("Практика №6 выполнена; результаты сохранены в", OUT)')
for cell in C:
    if cell.cell_type == 'code':
        cell.source = cell.source.replace('spark.createDataFrame(', 'local_frame(')
        cell.source = cell.source.replace('= PipelineModel.load(', '= load_pipeline(')
        cell.source = cell.source.replace('=PipelineModel.load(', '=load_pipeline(')
nb = nbf.v4.new_notebook(cells=C, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
nbf.write(nb, ROOT / 'Практика_6.ipynb')
print('Создано ячеек:', len(C))
