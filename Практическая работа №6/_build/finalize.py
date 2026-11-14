"""Отчёт по фактическим результатам и включение выводов в выполненный блокнот."""
from pathlib import Path
import csv
import json
from collections import Counter
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'practice6_results'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def table(frame):
    def fmt(x):
        if isinstance(x,float):return f'{x:.4f}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

sent=read('model_config.json')
sent_cov=read('sentiment_coverage.json')
lit=read('literature_test_metrics.json')
lit_cov=read('literature_coverage.json')
gen=read('literature_generation_config.json')
validation=pd.read_csv(OUT/'validation.csv')
lit_validation=pd.read_csv(OUT/'literature_validation.csv')
works=pd.read_csv(OUT/'literature_works.csv')
review=pd.read_csv(OUT/'literature_review.csv')
notes=json.loads((ROOT/'_build/error_notes.json').read_text(encoding='utf-8'))
sent_errors=pd.read_csv(OUT/'error_examples.csv')
lit_errors=pd.read_csv(OUT/'literature_error_examples.csv')
parts=[]
def add(s):parts.append(s.strip())
add('# Результаты практической работы №6\n\nДицель Н.С., ИКБО-11-23. Выполнено 02.11.2026. Python 3.12 / Spark 3.5.3 / Java 21.')
add('''## 1. Данные и контроль воспроизводимости

Из 290 458 CSV-записей отобраны все 86 436 rureviews: по 28 812 каждого класса. Точных повторов нет; один исходный отзыв состоит из пробела. После токенизации 31 запись без токенов, 84 709 групп, 298 групп с разными метками. Такие записи сохранены в оценке.

Группы одинаковых токенизированных текстов целиком отнесены к train/valid/test по SHA-256 и seed=42. Использованы фактические хеши, а не число строк из контрольной таблицы. Численность положительного класса в частях немного отличается от методички; независимый расчёт токенов и хешей в Python дал те же группы и разбиение, что Spark.

Словарь и IDF обучались только на train; варианты сравнивались по одному valid. Шаблон методички сохраняет отрицания и эмоциональную пунктуацию, но не эмодзи. Морфологическая нормализация не применялась. Ручной IDF на трёх документах совпал со Spark: ln((N+1)/(DF+1)).''')
split=pd.read_parquet(OUT/'split_manifest').set_index('row_id')
counts=Counter()
with (ROOT/'practice6_data/rureviews.jsonl').open(encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line)
        counts[(split.loc[row['row_id'],'split'],int(row['label']))]+=1
split_table=pd.DataFrame([dict(split=s,neutral=counts[(s,0)],positive=counts[(s,1)],negative=counts[(s,2)]) for s in ['train','valid','test']])
add(table(split_table))
add('## 2. Классификация тональности\n\nВсе значения ниже рассчитаны выполненным блокнотом. Постоянный нейтральный класс на valid: macro-F1=0,1669.')
add(table(validation[['model','vocabulary','fit_s','accuracy','macro_f1','log_loss']]))
add(f'''По valid выбрана **{sent['model']}**. На test: accuracy **{sent['test_metrics']['accuracy']:.4f}**, macro-F1 **{sent['test_metrics']['macro_f1']:.4f}**, weighted-F1 {sent['test_metrics']['weighted_f1']:.4f}, log loss {sent['test_metrics']['log_loss']:.4f}.

OOV по вхождениям признаковых токенов: **{sent_cov['oov_fraction']:.2%}**; нулевых векторов **{sent_cov['zero_vectors']}**, среднее nnz {sent_cov['mean_nnz']:.2f}. Нулевой вектор не удаляется ради улучшения метрик.

BoW LR превзошла нормированные TF–IDF варианты при заданной регуляризации. Масштаб представлений различается, поэтому результат не доказывает универсального превосходства BoW. Удаление отрицаний стандартным стоп-листом теряет смысловую информацию; выбранные варианты сравнивались по фактическому macro-F1, без использования test.''')
add(table(pd.read_csv(OUT/'test_by_class.csv')))
add('### Разбор ошибок тональности')
for row in sent_errors.head(5).itertuples():
    key=str(row.row_id)
    assert key in notes['sentiment']
    add(f"**row_id={key}: {int(row.label)} → {int(row.prediction)}, уверенность {row.confidence:.3f}.** «{row.text}»\n\n{notes['sentiment'][key]}")
add('## 3. Самостоятельная часть: Literature')
add('''Проверены 103 файла трёх авторов из prose. После очистки оставлено 94 самостоятельных произведения: 73 Чехова, 13 Гоголя, 8 Тургенева. Исключения и границы каждого файла перечислены в literature_review.csv. Пьесы, два сборника, фельетон, короткий опросник и проблемный отрывок исключены. У «Дуэли», «Драмы на охоте» и «Тараса Бульбы» удалены внутренние сноски, последующий основной текст сохранён.

«Мёртвые души» с обоими томами — одна группа. Точных повторов и совпадений проверяемых 50-словных блоков между выбранными произведениями не найдено. Проверка блоков не является исчерпывающим поиском всех перефразированных совпадений.

Разбиение по произведениям выполнено до нарезки. Фрагмент — 300 слов, шаг 300, хвост отброшен. До 50 train-фрагментов каждой книги отобраны по устойчивому хешу; valid и test не ограничивались. Все авторы присутствуют во всех частях. Проверены отсутствие пересечения work_group и одинаковых полных фрагментов между частями.''')
composition=works.groupby(['author','split']).agg(works=('work_group','nunique'),fragments=('full_fragments','sum'),train_used=('train_used','sum')).reset_index()
add(table(composition))
add('### Valid и выбор модели\n\nСловарь до 20 000, minDF=3, у LR λ=0,001. Для сравнения удаления служебных слов использован стандартный русский стоп-лист.')
add(table(lit_validation[['model','vocabulary','fit_s','macro_f1','accuracy','log_loss']]))
add(f"""Выбрана **{lit['selected_model']}**. На test по фрагментам macro-F1 **{lit['fragment']['macro_f1']:.4f}**, accuracy **{lit['fragment']['accuracy']:.4f}**. Постоянный автор ({lit['baseline_author']}): macro-F1 {lit['baseline_fragment']['macro_f1']:.4f}.

По произведениям macro-F1 **{lit['work']['macro_f1']:.4f}**, accuracy **{lit['work']['accuracy']:.4f}**; постоянное правило — macro-F1 {lit['baseline_work']['macro_f1']:.4f}. Вероятности усреднялись по всем фрагментам произведения, затем выбирался максимум.

Размер словаря {lit_cov['vocabulary']}; OOV **{lit_cov['oov_fraction']:.2%}**; нулевых векторов {lit_cov['zero_vectors']}; среднее nnz {lit_cov['mean_nnz']:.2f}. Число тестовых произведений каждого автора приведено в support таблицы ниже; малое число книг ограничивает точность оценки переноса на новые произведения.""")
add(table(pd.read_csv(OUT/'literature_test_work_classes.csv')))
add('### Пять конкретных ошибок авторства')
assert len(lit_errors)>=5
authors=lit['labels']
for row in lit_errors.head(5).itertuples():
    assert row.row_id in notes['literature']
    add(f"**{row.work_group}, слова с позиции {row.start}: {authors[int(row.label)]} → {authors[int(row.prediction)]}, уверенность {row.confidence:.3f}.**\n\nФрагмент: «{row.text[:330]}…»\n\n{notes['literature'][row.row_id]}")
coef=pd.read_csv(OUT/'literature_coefficient_contrast.csv')
chekhov=', '.join(coef.sort_values('contrast',ascending=False).head(8).term)
turgenev=', '.join(coef.sort_values('contrast').head(8).term)
add(f'''### Признаки авторства и ограничения

В LR BoW в пользу Чехова относительно Тургенева: **{chekhov}**. В пользу Тургенева: **{turgenev}**.

Имена и сюжетные слова могут запоминать книги вместо стиля. Разбиение по произведениям препятствует попаданию соседних фрагментов одной истории в test; сравнение стоп-листов проверяет роль служебной лексики. Окончания сохранены как возможный стилевой сигнал. Оформление очищено единообразно, но редакции, эпоха и тематика остаются смешивающими факторами. Коэффициенты не являются причинными эффектами авторства.''')
add('''Особенно показателен контраст «всё» и «все»: сохранённое различие е/ё может отражать редакционную традицию, а не авторский стиль. Это ограничивает интерпретацию даже при очистке заголовков; дополнительный опыт с ё→е помог бы проверить устойчивость.

На valid удаление стоп-слов ухудшило LR BoW: macro-F1 0,8737 → 0,8234. Служебная лексика полезна в этом опыте. NB выбрана по macro-F1, но её log loss выше, чем у LR: хорошая классификация не означает надёжно откалиброванные вероятности. Уверенные ошибки NB требуют осторожной интерпретации.

Проверка сохранённого словаря подтвердила отсутствие «чичиков», «чичикова», «урбенина» и «лаврецкий». Модель оценивала новые произведения без этих специфических имён. Идеальные 20/20 ответов по произведениям не заменяют проверку на большем числе книг: у Гоголя в test только три, у Тургенева — две.''')
add('## 4. Генерация текста')
def edge_stats(filename):
    edges=read(filename)
    rows=[]
    for order in [1,2]:
        subset=[r for r in edges if r['order']==order]
        rows.append(dict(order=order,states=len({r['context'] for r in subset}),edges=len(subset),observations=sum(r['count'] for r in subset)))
    return pd.DataFrame(rows)
add('Основная часть: цепи по 3 000 train-отзывов, отдельно от test. Счётчики распределённо рассчитаны Spark NGram и groupBy.')
add(table(edge_stats('markov_edges.json')))
add('Самостоятельная часть: полные train-произведения Чехова, предложения длиной 3–150 токенов. BOS/EOS задают границы. Локальная генерация использует таблицу частот, temperature=1, до 50 новых токенов.')
add(table(edge_stats('literature_markov_edges.json')))
generated=pd.read_csv(OUT/'literature_generations.csv')
add(table(generated[['order','seed','generated_tokens','stopped_by','backoffs','text']]))
add('### Контекст и проверки')
add(table(pd.read_csv(OUT/'literature_context_tests.csv')[['prompt','generated_tokens','stopped_by','backoffs','text']]))
add('''Проверены воспроизводимость одинаковых seed, сумма условных вероятностей, отсутствие перехода между учебными предложениями, остановка по EOS и max_tokens, backoff с пары к последнему слову и со неизвестного слова к BOS. Обе причины остановки также проверены на детерминированных цепях. Пример реальной остановки EOS сохранён в literature_generation_config.json.

Второй порядок удерживает более длинные сочетания и увеличивает число состояний. Первый чаще нарушает грамматику, склеивая частотные фрагменты. Ни одна цепь не моделирует цель высказывания, сюжет или дальнюю согласованность. Temperature меняет выбор среди наблюдавшихся продолжений, но не добавляет отсутствующие переходы.''')
add(table(pd.read_csv(OUT/'literature_generation_copy_check.csv')))
add('Совпадение целого предложения проверено относительно train. Отсутствие полного совпадения не исключает копирования отдельных частей; короткие распространённые фразы также могут совпадать естественно.')
add('''## 5. Что сохранено и проверено

Сохранены полные PipelineModel тональности и авторства, словари, IDF, JSON-паспорта, разбиения по строкам/произведениям/фрагментам, метрики, ошибки, графики и таблицы переходов в JSON/Parquet. После загрузки моделей классы и вероятности совпали: 4 новых отзыва и 5 фиксированных тестовых фрагментов.

Основная обработка и обучение выполнены в Spark. Малые локальные демонстрационные таблицы передаются через JSONL, nnz считается нативным выражением по разреженному вектору: обход сбоя Python worker в Windows не меняет алгоритмы MLlib. На драйвер переносятся агрегаты, небольшие примеры и ограниченные таблицы переходов.

Результаты относятся к заданным корпусам, очистке, группировке и разбиению. Для оценки устойчивости нужны дополнительные разбиения целых произведений; они не использовались для подбора по текущему test.''')
report='\n\n'.join(parts)+'\n'
(ROOT/'Результаты_практики_6.md').write_text(report,encoding='utf-8')
path=ROOT/'Практика_6.ipynb'
nb=nbformat.read(path,as_version=4)
nb.cells=[c for c in nb.cells if not (c.cell_type=='markdown' and c.metadata.get('practice6_report'))]
cell=nbformat.v4.new_markdown_cell('## 11. Фактические результаты и разбор ошибок\n\n'+report.split('\n',1)[1])
cell.metadata['practice6_report']=True
for i,c in enumerate(nb.cells):
    if c.cell_type=='markdown' and c.source.startswith('## 11. Итоговые выводы'):
        nb.cells[i]=cell;break
else:nb.cells.insert(-1,cell)
nbformat.validate(nb)
errors=[(i,o.get('ename')) for i,c in enumerate(nb.cells) for o in c.get('outputs',[]) if o.output_type=='error']
assert not errors,errors
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
nbformat.write(nb,path)
print('Отчёт сохранён и включён в блокнот; все ячейки выполнены без ошибок.')
