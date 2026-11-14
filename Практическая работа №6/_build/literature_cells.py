"""Ячейки самостоятельной части; исполняется сборщиком make_nb.py."""
md("""## 9. Самостоятельная работа: авторство и генерация на Literature

Классифицируем Чехова, Гоголя и Тургенева по 300-словным фрагментам новых произведений. Генерируем текст по train-произведениям Чехова.

В ZIP 373 TXT. Проверены границы 103 файлов выбранных авторов; включены 94: Чехов — 73, Гоголь — 13, Тургенев — 8. Исключены две пьесы Чехова, «Встреча весны», короткий опросник «Дополнительные вопросы» (85 слов), «Ревизор», «Вечера на хуторе…» (пересечение с «Ночью перед Рождеством»), «Ночи на вилле» (отрывок и комментарии), «Стихотворения в прозе» и «Записки охотника» (сборник).

Границы основного текста заданы срезами в literature_review.csv. Авторские строки, названия, посвящения, эпиграфы, номера глав и примечания удалены. В «Дуэли», «Драме на охоте» и «Тарасе Бульбе» удаляются внутренние сноски с сохранением следующего за ними текста. «Мёртвые души» с двумя томами — одна группа. Две «Беды» Чехова — разные рассказы 1886 и 1887 годов.

До разбиения проверяются точные хеши и общие 50-словные блоки. В паспорте сохранены решения, кодировки и границы. Восстановление CP437 относится к имени ZIP-записи, CP1251 — к содержимому известного исключения в полном инвентаре.""")
prep = (ROOT / '_build/prepare_literature.py').read_text(encoding='utf-8')
code(prep.replace("ROOT = Path(__file__).resolve().parent.parent", "# ROOT задан в первой ячейке"))
md("""### 9.1. Разбиение произведений и подготовка фрагментов

SHA-256 задаёт порядок групп внутри автора. Доли 60/20/20 с округлением вниз границ. Затем нарезаем непересекающиеся фрагменты по 300 слов; неполные хвосты отбрасываем. В train оставляем до 50 фрагментов каждой группы по устойчивому хешу. Valid и test сохраняют все фрагменты. Основной вариант сохраняет служебные слова и словоформы, регистр понижается.""")
code(r'''
AUTHORS = ["Chekhov", "Gogol", "Turgenev"]
lit_docs = spark.read.schema("document_id STRING, author STRING, filename STRING, work_group STRING, text STRING, body_chars LONG").json((WORK/"literature_selected.jsonl").as_posix())
units = lit_docs.select("author","work_group").distinct()
assert units.groupBy("author").count().filter("count < 6").count()==0
w=Window.partitionBy("author").orderBy(F.sha2("work_group",256))
wa=Window.partitionBy("author")
units=(units.withColumn("r",F.row_number().over(w)).withColumn("n",F.count("*").over(wa))
    .withColumn("split",F.when(F.col("r")<=F.floor(F.col("n")*.6),"train")
        .when(F.col("r")<=F.floor(F.col("n")*.8),"valid").otherwise("test"))
    .select("author","work_group","split"))
lit_docs=lit_docs.join(units,["author","work_group"]).cache()
wt=RegexTokenizer(inputCol="text",outputCol="words",pattern=r"(?U)[^\W_]+",gaps=False,minTokenLength=1)
word_docs=wt.transform(lit_docs)
fragments=(word_docs.filter(F.size("words")>=300)
    .withColumn("start",F.explode(F.sequence(F.lit(1),F.size("words")-299,F.lit(300))))
    .withColumn("fragment",F.expr("slice(words,start,300)")))
label_map=F.create_map(*[v for i,a in enumerate(AUTHORS) for v in (F.lit(a),F.lit(float(i)))])
fragments=(fragments.select("document_id","author","filename","work_group","split","start",
    F.concat_ws(" ","fragment").alias("text"),label_map[F.col("author")].alias("label"))
    .withColumn("row_id",F.sha2(F.concat_ws(":","document_id","start"),256)).cache())
assert fragments.groupBy("work_group").agg(F.countDistinct("split").alias("n")).filter("n>1").count()==0
assert fragments.groupBy("text").agg(F.countDistinct("split").alias("n")).filter("n>1").count()==0
assert fragments.select("row_id").distinct().count()==fragments.count()
assert fragments.select("author","split").distinct().count()==9
cw=Window.partitionBy("work_group").orderBy(F.sha2("row_id",256),"row_id")
lit_train=(fragments.filter("split='train'").withColumn("rank",F.row_number().over(cw))
    .filter("rank<=50").drop("rank").cache())
lit_valid=fragments.filter("split='valid'")
lit_test=fragments.filter("split='test'")
counts=fragments.groupBy("work_group").count().withColumnRenamed("count","full_fragments")
used=lit_train.groupBy("work_group").count().withColumnRenamed("count","train_used")
work_pdf=(lit_docs.select("author","filename","work_group","body_chars","split")
    .join(counts,"work_group","left").join(used,"work_group","left").fillna(0)
    .orderBy("author","work_group").toPandas())
work_pdf.to_csv(OUT/"literature_works.csv",index=False)
units.orderBy("author","work_group").toPandas().to_csv(OUT/"literature_split.csv",index=False)
fragments.select("row_id","document_id","work_group","start","split","label").write.mode("overwrite").parquet((OUT/"literature_fragment_manifest").as_posix())
print("Произведения:");units.groupBy("author","split").count().orderBy("author","split").show(30)
print("Фрагменты:");fragments.groupBy("author","split").count().orderBy("author","split").show(30)
print("Использовано train:");lit_train.groupBy("author").count().show()
display(work_pdf)
''')
md('### 9.2. Состав корпуса и частотные профили')
code(r'''
group_counts=units.groupBy("author","split").count().toPandas()
fragment_counts=fragments.groupBy("author","split").count().toPandas()
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,data,title,ylabel in [(axes[0],group_counts,"Произведения: train/valid/test","Число произведений"),
    (axes[1],fragment_counts,"Полные фрагменты: train/valid/test","Число фрагментов")]:
    data.pivot(index="author",columns="split",values="count").reindex(columns=["train","valid","test"]).plot.bar(ax=ax)
    ax.set(title=title,xlabel="Автор",ylabel=ylabel);ax.tick_params(axis="x",rotation=0)
fig.tight_layout();fig.savefig(OUT/"lit_01_composition.png",dpi=180);plt.show()
lit_occ=text_prep.transform(lit_train).select("author","row_id",F.explode("tokens").alias("term"))
terms=[r.term for r in lit_occ.groupBy("term").count().filter(~F.col("term").isin(RU_STOP)).orderBy(F.desc("count"),"term").limit(16).collect()]
totals=lit_occ.groupBy("author").count().withColumnRenamed("count","total")
profile=(lit_occ.filter(F.col("term").isin(terms)).groupBy("author","term").count().join(totals,"author")
    .withColumn("per1000",1000*F.col("count")/F.col("total")).select("author","term","per1000").toPandas()
    .pivot(index="term",columns="author",values="per1000").fillna(0).reindex(terms))
fig,ax=plt.subplots(figsize=(9,5));profile.plot.barh(ax=ax)
ax.set(xlabel="Вхождения на 1 000 токенов train",ylabel="Токен",title="Авторы: train после ограничения 50 фрагментов")
fig.tight_layout();fig.savefig(OUT/"lit_02_word_profiles.png",dpi=180);plt.show()
profile.to_csv(OUT/"literature_word_profile.csv")
''')
md('**Вывод:** произведений Чехова больше, но романы Тургенева длиннее рассказов. Ограничение train сдерживает влияние длинных книг. Оценка по произведениям даёт каждой книге один голос. Частоты нормированы числом токенов автора.')
md("""### 9.3. Обучение и выбор по valid

Постоянное правило — наиболее частый автор использованного train. Сравниваем NB BoW, LR BoW, LR TF–IDF и LR BoW без русского стоп-листа. Последний опыт проверяет роль служебных слов. Параметры моделей те же, что в основном опыте; test не участвует в выборе.""")
code(r'''
lit_majority=float(lit_train.groupBy("label").count().orderBy(F.desc("count"),"label").first().label)
lit_baseline_valid,_,_=multiclass_metrics(lit_valid.withColumn("prediction",F.lit(lit_majority)))
lit_models,lit_rows={},[]
for name,base,stop in [("NB_BOW","NB_BOW",None),("LR_BOW","LR_BOW",None),("LR_TFIDF","LR_TFIDF",None),("LR_BOW_DEFAULT_STOP","LR_BOW",RU_STOP)]:
    t0=perf_counter();model=make_pipeline(base,stop_words=stop).fit(lit_train);fit_s=perf_counter()-t0
    pred=model.transform(lit_valid).cache()
    metrics,byclass,_=multiclass_metrics(pred)
    cv=next(s for s in model.stages if isinstance(s,CountVectorizerModel))
    lit_rows.append(dict(model=name,vocabulary=len(cv.vocabulary),fit_s=fit_s,**metrics))
    byclass["author"]=AUTHORS;byclass.to_csv(OUT/f"literature_valid_{name}_classes.csv",index=False)
    pred.unpersist();lit_models[name]=model
lit_comparison=pd.DataFrame(lit_rows).sort_values(["macro_f1","model"],ascending=[False,True])
lit_comparison.to_csv(OUT/"literature_validation.csv",index=False)
print("Постоянный автор:",AUTHORS[int(lit_majority)],lit_baseline_valid);display(lit_comparison.round(4))
fig,ax=plt.subplots(figsize=(8,4));ax.bar(lit_comparison.model,lit_comparison.macro_f1)
ax.axhline(lit_baseline_valid["macro_f1"],ls="--",label="Постоянный автор")
ax.set(ylabel="Macro-F1",title="Выбор модели авторства по valid",ylim=(0,1));ax.tick_params(axis="x",rotation=15);ax.legend()
fig.tight_layout();fig.savefig(OUT/"lit_03_validation.png",dpi=180);plt.show()
''')
md('### 9.4. Test: фрагменты и произведения\n\nУсредняем вероятности фрагментов произведения, затем берём argmax. Номер автора заранее задан списком AUTHORS и не входит в признаки.')
code(r'''
LIT_BEST_NAME=str(lit_comparison.iloc[0].model);LIT_BEST=lit_models[LIT_BEST_NAME]
lit_pred=LIT_BEST.transform(lit_test).cache()
lit_test_metrics,lit_test_classes,lit_cm=multiclass_metrics(lit_pred)
lit_base_test,_,_=multiclass_metrics(lit_test.withColumn("prediction",F.lit(lit_majority)))
p=lit_pred.withColumn("p",vector_to_array("probability"))
lit_work_pred=(p.groupBy("work_group","label").agg(*[F.avg(F.col("p")[k]).alias(f"p{k}") for k in range(3)],F.count("*").alias("fragments"))
    .withColumn("p",F.array("p0","p1","p2"))
    .withColumn("prediction",(F.expr("array_position(p,array_max(p))")-1).cast("double")).cache())
lit_work_metrics,lit_work_classes,lit_work_cm=multiclass_metrics(lit_work_pred)
lit_work_baseline,_,_=multiclass_metrics(lit_work_pred.withColumn("prediction",F.lit(lit_majority)))
print("Выбрана:",LIT_BEST_NAME)
print("Test фрагменты:",lit_test_metrics,"baseline:",lit_base_test)
print("Test произведения:",lit_work_metrics,"baseline:",lit_work_baseline)
lit_test_classes["author"]=AUTHORS;lit_work_classes["author"]=AUTHORS
display(lit_test_classes.round(4));display(lit_work_classes.round(4))
lit_work_pred.orderBy("work_group").toPandas().drop(columns="p").to_csv(OUT/"literature_work_predictions.csv",index=False)
lit_test_classes.to_csv(OUT/"literature_test_fragment_classes.csv",index=False)
lit_work_classes.to_csv(OUT/"literature_test_work_classes.csv",index=False)
lit_result=dict(selected_model=LIT_BEST_NAME,labels=AUTHORS,baseline_author=AUTHORS[int(lit_majority)],fragment=lit_test_metrics,
    work=lit_work_metrics,baseline_fragment=lit_base_test,baseline_work=lit_work_baseline,test_work_support=lit_work_classes.support.tolist())
(OUT/"literature_test_metrics.json").write_text(json.dumps(lit_result,ensure_ascii=False,indent=2),encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,cm,title in [(axes[0],lit_cm,"Test: фрагменты"),(axes[1],lit_work_cm,"Test: произведения")]:
    ax.imshow(cm/np.maximum(cm.sum(axis=1,keepdims=True),1),vmin=0,vmax=1,alpha=.55)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    ax.set_xticks(range(3),AUTHORS);ax.set_yticks(range(3),AUTHORS)
    ax.set(xlabel="Предсказанный автор",ylabel="Истинный автор",title=title)
fig.tight_layout();fig.savefig(OUT/"lit_04_confusion.png",dpi=180);plt.show()
''')
md('### 9.5. OOV, nnz, ошибки и различающие слова\n\nСохраняем ошибки с названием, позицией, исходным текстом и уверенностью. Сначала берём ошибки из разных книг. Контраст LR BoW: Чехов против Тургенева.')
code(r'''
lit_cv=next(s for s in LIT_BEST.stages if isinstance(s,CountVectorizerModel))
vocab=spark.createDataFrame([(t,) for t in lit_cv.vocabulary],["term"])
lit_terms=lit_pred.select(F.explode(lit_cv.getInputCol()).alias("term"))
lit_oov=lit_terms.join(F.broadcast(vocab),"term","left_anti").count()/max(lit_terms.count(),1)
lit_cov=lit_pred.withColumn("nnz",nnz("features"))
lit_coverage=dict(vocabulary=len(lit_cv.vocabulary),oov_fraction=lit_oov,zero_vectors=lit_cov.filter("nnz=0").count(),
    mean_nnz=float(lit_cov.agg(F.avg("nnz")).first()[0]))
print(lit_coverage);(OUT/"literature_coverage.json").write_text(json.dumps(lit_coverage,indent=2),encoding="utf-8")
err=(lit_pred.filter("label!=prediction").withColumn("confidence",F.array_max(vector_to_array("probability")))
    .withColumn("error_rank",F.row_number().over(Window.partitionBy("work_group").orderBy(F.desc("confidence"),"row_id"))))
err_pdf=(err.orderBy("error_rank",F.desc("confidence"),"row_id")
    .select("row_id","work_group","start","label","prediction","confidence","text").limit(8).toPandas())
err_pdf.to_csv(OUT/"literature_error_examples.csv",index=False)
for r in err_pdf.itertuples():
    print(r.work_group,r.start,AUTHORS[int(r.label)],"→",AUTHORS[int(r.prediction)],round(r.confidence,4));print(r.text[:650],"\n")
lr_lit=lit_models["LR_BOW"]
lr_lit_cv=next(s for s in lr_lit.stages if isinstance(s,CountVectorizerModel))
weights=lr_lit.stages[-1].coefficientMatrix.toArray();delta=weights[0]-weights[2]
idx=np.r_[np.argsort(delta)[:12],np.argsort(delta)[-12:]]
lit_coef=pd.DataFrame(dict(term=np.array(lr_lit_cv.vocabulary)[idx],contrast=delta[idx])).sort_values("contrast")
lit_coef.to_csv(OUT/"literature_coefficient_contrast.csv",index=False)
fig,ax=plt.subplots(figsize=(8,7));ax.barh(lit_coef.term,lit_coef.contrast);ax.axvline(0,lw=1)
ax.set(xlabel="w(Чехов) − w(Тургенев)",ylabel="Токен",title="LR BoW, train: различающие слова")
fig.tight_layout();fig.savefig(OUT/"lit_05_coefficients.png",dpi=180);plt.show()
''')
md('**Вывод:** имена и сюжетная лексика характеризуют конкретные книги. Служебные слова и окончания могут отражать стиль. Заголовки и примечания удалены; обобщение на другие жанры не проверялось. Конкретные ошибки разобраны в итоговом отчёте.')
md('### 9.6. Сохранение полного PipelineModel и проверка загрузки')
code(r'''
lit_path=OUT/"literature_pipeline";LIT_BEST.write().overwrite().save(lit_path.as_posix())
lit_loaded=PipelineModel.load(lit_path.as_posix())
probe=lit_test.orderBy("row_id").limit(5).cache()
old=LIT_BEST.transform(probe).orderBy("row_id").select("prediction","probability").collect()
new=lit_loaded.transform(probe).orderBy("row_id").select("prediction","probability").collect()
assert all(x.prediction==y.prediction and np.allclose(x.probability,y.probability) for x,y in zip(old,new))
lit_config=dict(model=LIT_BEST_NAME,authors=AUTHORS,seed=SEED,fragment_words=300,stride=300,train_cap_per_work=50,
    split="sha256(work_group), per-author 60/20/20; floor boundaries",token_pattern=TOKEN_PATTERN,min_df=MIN_DF,
    max_vocab=VOCAB_SIZE,fitted_vocab=len(lit_cv.vocabulary),reg_param=REG_PARAM,spark_version=spark.version,
    python_version=platform.python_version(),test=lit_result,coverage=lit_coverage)
(OUT/"literature_model_config.json").write_text(json.dumps(lit_config,ensure_ascii=False,indent=2),encoding="utf-8")
print("После загрузки совпадают вероятности и метки 5 фрагментов.");probe.unpersist()
''')
md("""### 9.7. Генерация по полным train-произведениям Чехова

Используем основные тексты до ограничения числа фрагментов. Словоформы, служебные слова и пунктуация сохранены. Предложения выделяются внутри произведений; длина 3–150 токенов. Таблицы Spark NGram сохраняются в Parquet и JSON. До локального переноса проверяется предел 1 000 000 переходов. Для каждого порядка получаем три примера при seed 7, 42, 101.""")
code(r'''
GEN_AUTHOR="Chekhov"
lit_sent=(lit_docs.filter((F.col("author")==GEN_AUTHOR)&(F.col("split")=="train"))
    .select("document_id",F.posexplode(F.split("text",r"(?<=[.!?…])\s+|[\r\n]+")).alias("sentence_id","sentence")))
lit_sent=markov_tokenizer.transform(lit_sent).filter((F.size("words")>=3)&(F.size("words")<=150)).cache()
lit_edges=count_transitions(lit_sent,1).unionByName(count_transitions(lit_sent,2)).cache()
assert lit_edges.limit(1_000_001).count()<=1_000_000
lit_edges.write.mode("overwrite").parquet((OUT/"literature_markov_edges").as_posix())
lit_edge_rows=[r.asDict() for r in lit_edges.collect()]
(OUT/"literature_markov_edges.json").write_text(json.dumps(lit_edge_rows,ensure_ascii=False),encoding="utf-8")
lit_chain=build_chain(lit_edge_rows)
print("Train-предложений:",lit_sent.count())
lit_edges.groupBy("order").agg(F.count("*").alias("edges"),F.countDistinct("context").alias("states"),
    F.sum("count").alias("observed_transitions")).orderBy("order").show()
gen_rows=[]
for order in (1,2):
    for seed in (7,42,101):
        result=generate(lit_chain,order=order,seed=seed,max_tokens=50)
        assert result==generate(lit_chain,order=order,seed=seed,max_tokens=50)
        gen_rows.append(dict(order=order,seed=seed,**result));print(order,seed,result)
pd.DataFrame(gen_rows).to_csv(OUT/"literature_generations.csv",index=False)
''')
md('### 9.8. Контексты, вероятности и проверки генератора')
code(r'''
known=(lit_edges.filter((F.col("order")==2)&(~F.col("context").contains("<BOS>"))).groupBy("context")
    .agg(F.sum("count").alias("n")).orderBy(F.desc("n"),"context").first().context)
context_tests=[]
for prompt in (known,"квазигиперсловонет","квазигиперсловонет он"):
    result=generate(lit_chain,prompt=prompt,order=2,seed=42,max_tokens=30)
    context_tests.append(dict(prompt=prompt,**result));print(prompt,result)
assert context_tests[1]["backoffs"]>=1
pd.DataFrame(context_tests).to_csv(OUT/"literature_context_tests.csv",index=False)
options=lit_chain[(1,("он",))];total=sum(c for _,c in options)
assert math.isclose(sum(c/total for _,c in options),1.)
top=sorted(options,key=lambda x:(-x[1],x[0]))[:15]
fig,ax=plt.subplots(figsize=(8,5));ax.barh([w for w,_ in top][::-1],[c/total for _,c in top][::-1])
ax.set(xlabel="P(следующий токен | он)",ylabel="Продолжение",title="Чехов, train: первый порядок")
fig.tight_layout();fig.savefig(OUT/"lit_06_markov_next.png",dpi=180);plt.show()
boundary_df=spark.createDataFrame([(["а","б"],),(["в","г"],)],["words"])
boundary_edges=count_transitions(boundary_df,1).collect()
assert not any(r.context=="б" and r.next=="в" for r in boundary_edges)
assert sum(r["count"] for r in boundary_edges if r.next=="<EOS>")==2
eos_chain={(1,("<BOS>",)):[("<EOS>",1)]}
loop_chain={(1,("<BOS>",)):[("а",1)],(1,("а",)):[("а",1)]}
assert generate(eos_chain,order=1)["stopped_by"]=="eos"
assert generate(loop_chain,order=1,max_tokens=7)["generated_tokens"]==7
assert generate(loop_chain,order=1,max_tokens=7)["stopped_by"]=="max_tokens"
short_chain={(2,("<BOS>","<BOS>")):[("он",1)],(1,("он",)):[("<EOS>",1)]}
assert generate(short_chain,prompt="несуществующий он",order=2)["backoffs"]==1
real_eos=None
for s in range(100):
    result=generate(lit_chain,order=2,seed=s,max_tokens=150)
    if result["stopped_by"]=="eos":real_eos=dict(seed=s,**result);break
assert real_eos is not None
print("Реальная остановка EOS:",real_eos)
copy_checks=[]
sentence_text=lit_sent.withColumn("joined",F.concat_ws(" ","words"))
for row in gen_rows:
    joined=" ".join(re.findall(MARKOV_PATTERN,row["text"].lower()))
    copied=sentence_text.filter(F.col("joined")==joined).limit(1).count()>0
    copy_checks.append(dict(order=row["order"],seed=row["seed"],exact_train_sentence=copied))
pd.DataFrame(copy_checks).to_csv(OUT/"literature_generation_copy_check.csv",index=False)
gen_config=dict(author=GEN_AUTHOR,source="full train works",orders=[1,2],seeds=[7,42,101],temperature=1,max_tokens=50,
    sentence_tokens=[3,150],backoff="pair -> last token -> BOS",real_eos=real_eos,
    checks=dict(reproducible=True,boundaries=True,eos=True,max_tokens=True,backoff=True),copy_checks=copy_checks)
(OUT/"literature_generation_config.json").write_text(json.dumps(gen_config,ensure_ascii=False,indent=2),encoding="utf-8")
print("Проверены границы, seed, backoff, EOS и max_tokens.")
lit_pred.unpersist();lit_work_pred.unpersist();lit_train.unpersist();fragments.unpersist()
lit_docs.unpersist();lit_sent.unpersist();lit_edges.unpersist()
''')
md('**Вывод:** второй порядок сохраняет более длинные локальные сочетания и требует больше состояний; первый легче смешивает части разных фраз. Оба порядка не обеспечивают сюжет или логическую согласованность. Проверка совпадения целого предложения с train не исключает копирование отдельных отрывков.')
