import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("# Практическая работа №4. Исследование выполнения Spark-задач\nJobs/Stages/Tasks, partitions, shuffle, JOIN-стратегии, cache/persist, small files. Данные: `ecommerce.sqlite`.")

md("## 3.1 Окружение и паспорт стенда")
code('''
import os, sys, platform, subprocess, json, time, shutil
# Spark на Windows для записи Parquet требует winutils/hadoop.dll
if os.name == "nt" and os.path.isdir(r"C:\\hadoop\\bin"):
    os.environ["HADOOP_HOME"] = r"C:\\hadoop"
    os.environ["PATH"] = r"C:\\hadoop\\bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
import pyspark
R = {}   # сюда собираются результаты для отчёта
java = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
def mem_gb():
    import ctypes
    class M(ctypes.Structure):
        _fields_=[("l",ctypes.c_ulong),("ml",ctypes.c_ulong),("tp",ctypes.c_ulonglong),("ap",ctypes.c_ulonglong),
                  ("tpf",ctypes.c_ulonglong),("apf",ctypes.c_ulonglong),("tv",ctypes.c_ulonglong),("av",ctypes.c_ulonglong),("ae",ctypes.c_ulonglong)]
    m=M(); m.l=ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return round(m.tp/1024**3,1)
R["stand"] = {"ОС": platform.platform(), "Python": sys.version.split()[0], "Java": java,
              "PySpark / Spark": pyspark.__version__, "Логических CPU": os.cpu_count(),
              "Оперативная память": f"{mem_gb()} ГБ"}
print(json.dumps(R["stand"], ensure_ascii=False, indent=1))
''')

md("## 3.2 Staging: экспорт SQLite в CSV\nДополнительно к пяти таблицам основной части экспортируются `orders` и `order_payments` для самостоятельной части (вариант C).")
code('''
from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("ecommerce.sqlite")
STAGING = Path("spark_input"); STAGING.mkdir(exist_ok=True)
TABLES = ["geolocation","order_items","products","sellers","product_category_name_translation"]
SELF_TABLES = ["orders","order_payments"]

with sqlite3.connect(DB_PATH) as con:
    for table in TABLES + SELF_TABLES:
        out = STAGING / f"{table}.csv"
        first, total = True, 0
        for chunk in pd.read_sql_query(f"SELECT * FROM {table}", con, chunksize=100_000):
            chunk.to_csv(out, index=False, mode="w" if first else "a", header=first, encoding="utf-8")
            total += len(chunk); first = False
        print(f"{table:40s} {total:>9d} rows")
''')

md("## 3.3 SparkSession (AQE выключен, broadcast выключен, shuffle.partitions=8)")
code('''
from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel
from time import perf_counter
from statistics import median

spark = (SparkSession.builder.master("local[*]").appName("Practice4_SparkExecution")
    .config("spark.sql.shuffle.partitions","8")
    .config("spark.sql.adaptive.enabled","false")
    .config("spark.sql.autoBroadcastJoinThreshold","-1")
    .config("spark.ui.showConsoleProgress","false")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext
print("Master:", sc.master); print("Default parallelism:", sc.defaultParallelism)
print("Shuffle partitions:", spark.conf.get("spark.sql.shuffle.partitions"))
print("AQE:", spark.conf.get("spark.sql.adaptive.enabled")); print("Spark UI:", sc.uiWebUrl)
R["stand"]["spark.master"]=sc.master; R["stand"]["spark.default.parallelism"]=sc.defaultParallelism
R["stand"]["spark.sql.shuffle.partitions"]=spark.conf.get("spark.sql.shuffle.partitions")
R["stand"]["spark.sql.adaptive.enabled"]=spark.conf.get("spark.sql.adaptive.enabled")

import urllib.request
def api(path):
    with urllib.request.urlopen(f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/{path}") as r:
        return json.load(r)
def ui_summary(group):
    """Jobs / Stages / Tasks / Shuffle по job group из REST API Spark UI."""
    jobs = [j for j in api("jobs") if j.get("jobGroup")==group]
    ids = {s for j in jobs for s in j["stageIds"]}
    st = {s["stageId"]: s for s in api("stages") if s["stageId"] in ids}
    key = max(st.values(), key=lambda s: s["numTasks"]) if st else None
    return {"Jobs": len(jobs), "Stages": len(ids),
            "Tasks по Stages": [st[i]["numTasks"] for i in sorted(st)],
            "Tasks в ключевой Stage": key["numTasks"] if key else None,
            "Shuffle Write, KiB": round(sum(s["shuffleWriteBytes"] for s in st.values())/1024,1),
            "Shuffle Read, KiB": round(sum(s["shuffleReadBytes"] for s in st.values())/1024,1)}
''')

md("## 3.4 CSV → Parquet")
code('''
PARQUET = Path("spark_input_parquet"); PARQUET.mkdir(exist_ok=True)
for name in TABLES + SELF_TABLES:
    df = spark.read.option("header",True).option("inferSchema",True).csv(str(STAGING/f"{name}.csv"))
    df.write.mode("overwrite").parquet(str(PARQUET/name))

geo = spark.read.parquet(str(PARQUET/"geolocation"))
items = spark.read.parquet(str(PARQUET/"order_items"))
products = spark.read.parquet(str(PARQUET/"products"))
sellers = spark.read.parquet(str(PARQUET/"sellers"))
translations = spark.read.parquet(str(PARQUET/"product_category_name_translation"))
print(geo.count(), items.count(), products.count(), sellers.count())
''')

md("## 4. Spark UI: маркировка")
code('''
sc.setJobGroup("P4_BASELINE","Практика 4: базовая проверка Spark UI",interruptOnCancel=True)
_ = items.count()
R["baseline"] = ui_summary("P4_BASELINE"); print(R["baseline"])
''')

md("## 5. Ленивые вычисления")
code('''
sc.setJobGroup("P4_LAZY","Lazy evaluation")
t0 = perf_counter()
geo_prepared = (geo.select("geolocation_zip_code_prefix","geolocation_state","geolocation_lat","geolocation_lng")
    .filter(F.col("geolocation_state").isNotNull())
    .withColumn("abs_lat", F.abs(F.col("geolocation_lat"))))
t_build = perf_counter()-t0
jobs_mid = len([j for j in api("jobs") if j.get("jobGroup")=="P4_LAZY"])
print(type(geo_prepared).__name__, f"построение цепочки: {t_build:.4f} с; Jobs в группе: {jobs_mid}")

t0 = perf_counter(); n = geo_prepared.count(); elapsed = perf_counter()-t0
jobs_after = len([j for j in api("jobs") if j.get("jobGroup")=="P4_LAZY"])
print("Rows:", n); print(f"Elapsed: {elapsed:.3f} s; Jobs в группе после count(): {jobs_after}")
R["lazy"] = {"build_s": t_build, "jobs_after_transformations": jobs_mid, "count_s": elapsed, "rows": n, "jobs_after_action": jobs_after}
''')

md("## 6. Партиции и параллелизм")
code('''
R["input_partitions"] = geo.rdd.getNumPartitions()
print("Input partitions:", R["input_partitions"]); print("Default parallelism:", sc.defaultParallelism)
sc.setJobGroup("P4_PARTS","Распределение строк по входным партициям")
partition_sizes = geo.select(F.spark_partition_id().alias("pid")).groupBy("pid").count().orderBy("pid")
partition_sizes.show(100, truncate=False)
R["parts_ui"] = ui_summary("P4_PARTS")
''')
code('''
state_counts = geo.groupBy("geolocation_state").count().orderBy(F.desc("count"))
state_counts.show(5, truncate=False)
tot = geo.count()
R["top_states"] = [(r[0], r[1], round(r[1]/tot*100,2)) for r in state_counts.take(5)]
''')

md("## 7. repartition() vs coalesce()")
code('''
skewed16 = geo.repartition(16, "geolocation_state")
repart4 = skewed16.repartition(4)
coal4 = skewed16.coalesce(4)
print("skewed16:", skewed16.rdd.getNumPartitions(), "repart4:", repart4.rdd.getNumPartitions(), "coal4:", coal4.rdd.getNumPartitions())
print("=== repartition(4) ==="); repart4.explain("formatted")
print("=== coalesce(4) ==="); coal4.explain("formatted")
''')
code('''
def partition_profile(df):
    return df.select(F.spark_partition_id().alias("pid")).groupBy("pid").count().orderBy("pid")
R["profile"] = {}
for name, df, g in [("repartition(4)", repart4, "P4_REPART"), ("coalesce(4)", coal4, "P4_COALESCE")]:
    sc.setJobGroup(g, name)
    print(name); prof = partition_profile(df); prof.show()
    cnt = [r["count"] for r in prof.collect()]
    R["profile"][name] = {"sizes": cnt, "min": min(cnt), "max": max(cnt), "mean": sum(cnt)/len(cnt), **ui_summary(g)}
print(R["profile"])
''')

md("## 8. groupBy и shuffle")
code('''
sc.setJobGroup("P4_GROUPBY","Shuffle: groupBy seller_id")
seller_stats = (items.groupBy("seller_id")
    .agg(F.count("*").alias("items"), F.round(F.sum("price"),2).alias("revenue"), F.round(F.avg("price"),2).alias("avg_price"))
    .orderBy(F.desc("revenue")))
seller_stats.show(5, truncate=False)
R["top_sellers"] = [tuple(r) for r in seller_stats.take(5)]
R["ui_groupby"] = ui_summary("P4_GROUPBY"); print(R["ui_groupby"])
seller_stats.explain("formatted")
''')

md("## 9. Чтение плана")
code('''
plan_demo = (items.filter(F.col("price")>=100).select("seller_id","price","freight_value")
    .withColumn("gross", F.col("price")+F.col("freight_value"))
    .groupBy("seller_id").agg(F.round(F.sum("gross"),2).alias("gross_sum")))
plan_demo.explain("formatted")
''')

md("## 10. Sort Merge Join vs Broadcast Hash Join")
code('''
import io, contextlib
def plan_text(df):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): df.explain("formatted")
    return buf.getvalue()

sc.setJobGroup("P4_SMJ","Sort Merge Join: items x products")
smj = items.hint("merge").join(products.hint("merge"), on="product_id", how="inner")
smj.explain("formatted")
t0=perf_counter(); n_smj = smj.count(); t_smj = perf_counter()-t0
print("Rows:", n_smj, f"{t_smj:.3f} s")

sc.setJobGroup("P4_BHJ","Broadcast Hash Join: items x products")
bhj = items.join(F.broadcast(products), on="product_id", how="inner")
bhj.explain("formatted")
t0=perf_counter(); n_bhj = bhj.count(); t_bhj = perf_counter()-t0
print("Rows:", n_bhj, f"{t_bhj:.3f} s")

ptxt_s, ptxt_b = plan_text(smj), plan_text(bhj)
R["join"] = {
  "SMJ": {"rows": n_smj, "time": t_smj, "exchange": ptxt_s.count(") Exchange"), "sorts": ptxt_s.count(") Sort"), "has_SortMergeJoin": "SortMergeJoin" in ptxt_s, **ui_summary("P4_SMJ")},
  "BHJ": {"rows": n_bhj, "time": t_bhj, "exchange": ptxt_b.count(") Exchange"), "has_BroadcastHashJoin": "BroadcastHashJoin" in ptxt_b, "has_BroadcastExchange": "BroadcastExchange" in ptxt_b, **ui_summary("P4_BHJ")}}
print(json.dumps(R["join"], ensure_ascii=False, indent=1))
''')

md("## 11. cache() / persist()")
code('''
spark.catalog.clearCache()
products_small = products.select("product_id","product_category_name")
sellers_small = sellers.select("seller_id","seller_state")
joined = (items.join(F.broadcast(products_small),"product_id","left")
    .join(F.broadcast(sellers_small),"seller_id","left")
    .withColumn("gross_value", F.col("price")+F.col("freight_value")))
def action_category(df):
    return df.groupBy("product_category_name").agg(F.sum("gross_value").alias("gross")).agg(F.sum("gross").alias("checksum")).first()[0]
def action_state(df):
    return df.groupBy("seller_state").agg(F.sum("gross_value").alias("gross")).agg(F.sum("gross").alias("checksum")).first()[0]

_ = action_category(joined)   # прогрев JVM (не входит в замеры)
sc.setJobGroup("P4_NOCACHE","Два действия без сохранения")
t0=perf_counter(); c1=action_category(joined); t1=perf_counter(); c2=action_state(joined); t2=perf_counter()
print(f"Action 1: {t1-t0:.3f} s\\nAction 2: {t2-t1:.3f} s\\nis_cached:", joined.is_cached)
R["nocache"] = {"a1": t1-t0, "a2": t2-t1, **ui_summary("P4_NOCACHE")}

sc.setJobGroup("P4_CACHE","persist MEMORY_AND_DISK + повторные Actions")
cached = joined.persist(StorageLevel.MEMORY_AND_DISK)
print("После persist:", cached.is_cached)
t0=perf_counter(); rows=cached.count(); t1=perf_counter()
k1=action_category(cached); t2=perf_counter()
k2=action_state(cached); t3=perf_counter()
storage = api("storage/rdd")
R["cache"] = {"mat": t1-t0, "a1": t2-t1, "a2": t3-t2, "rows": rows, "checksum_equal": abs(c1-k1)<1e-6 and abs(c2-k2)<1e-6,
              "storage": [{"name": s["name"][:60], "memory_kib": round(s["memoryUsed"]/1024,1), "disk_kib": round(s["diskUsed"]/1024,1), "partitions": s["numCachedPartitions"]} for s in storage],
              **ui_summary("P4_CACHE")}
print(f"Materialization: {t1-t0:.3f} s\\nCached action 1: {t2-t1:.3f} s\\nCached action 2: {t3-t2:.3f} s\\nRows:", rows)
cached.unpersist(blocking=True)
print("После unpersist:", cached.is_cached, "| Storage:", len(api("storage/rdd")))
''')

md("## 12. Число партиций и время выполнения")
code('''
N = 2_000_000
P_VALUES = [1,2,4,8,16,32]
spark.conf.set("spark.sql.shuffle.partitions","8")
def run_benchmark(p, repeats=3):
    times, checksum = [], None
    for _ in range(repeats):
        df = (spark.range(0,N,1,numPartitions=p)
              .withColumn("key",(F.col("id")%1000).cast("int"))
              .withColumn("value",F.sqrt(F.col("id")+1.0)))
        t0=perf_counter()
        checksum = df.groupBy("key").agg(F.sum("value").alias("s")).agg(F.sum("s").alias("checksum")).first()[0]
        times.append(perf_counter()-t0)
    return times, median(times), checksum
_ = run_benchmark(4, repeats=1)
bench = []
for p in P_VALUES:
    times, med, cs = run_benchmark(p)
    bench.append({"p":p,"runs":times,"median":med,"checksum":cs}); print(p, [round(t,3) for t in times], round(med,3), cs)
for b in bench: b["speedup"] = bench[0]["median"]/b["median"]
R["bench"] = bench
pd.DataFrame([{"p":b["p"],**{f"Run {i+1}":round(t,3) for i,t in enumerate(b["runs"])},"Median":round(b["median"],3),"S(p)":round(b["speedup"],2)} for b in bench])
''')
code('''
geo8 = geo.repartition(8)
R["shuffle_q"] = []
for q in [2,4,8,16,32]:
    spark.conf.set("spark.sql.shuffle.partitions", str(q))
    t0=perf_counter(); rows = geo8.groupBy("geolocation_state").count().count(); el = perf_counter()-t0
    print(f"q={q:2d}; groups={rows}; time={el:.3f}s"); R["shuffle_q"].append({"q":q,"groups":rows,"time":el})
spark.conf.set("spark.sql.shuffle.partitions","8")
''')

md("## 13. Партиции и маленькие файлы")
code('''
OUT_SMALL = Path("small_files_experiment"); OUT_SMALL.mkdir(exist_ok=True)
def write_and_measure(df, base, p):
    path = OUT_SMALL / f"{base}_{p}"
    df.repartition(p).write.mode("overwrite").parquet(str(path))
    files = list(path.glob("part-*.parquet")); sizes = [f.stat().st_size for f in files]
    return len(files), sum(sizes)/1024**2, (sum(sizes)/len(sizes)/1024) if sizes else 0
R["small"] = []
for p in [1,4,16,64]:
    r = write_and_measure(items, "p", p); R["small"].append({"p":p,"files":r[0],"total_mib":r[1],"avg_kib":r[2]}); print(p, r)
single = items.coalesce(1)
print("Partitions before write:", single.rdd.getNumPartitions())
single.write.mode("overwrite").parquet(str(OUT_SMALL/"coalesce_1"))
''')

md("## 14. AQE")
code('''
spark.conf.set("spark.sql.adaptive.enabled","true"); spark.conf.set("spark.sql.shuffle.partitions","32")
sc.setJobGroup("P4_AQE","groupBy с AQE")
adaptive = geo.groupBy("geolocation_state").count()
adaptive.explain("formatted"); adaptive.collect()
R["aqe"] = {"adaptive_plan": "AdaptiveSparkPlan" in plan_text(adaptive), **ui_summary("P4_AQE")}
print("AQE:", spark.conf.get("spark.sql.adaptive.enabled"), R["aqe"])
spark.conf.set("spark.sql.adaptive.enabled","false"); spark.conf.set("spark.sql.shuffle.partitions","8")
''')

md("""## 17. Самостоятельная работа: вариант C — `orders` × `order_payments`
Ключ `order_id`, группировка по `payment_type`.""")
code('''
orders = spark.read.parquet(str(PARQUET/"orders")); pay = spark.read.parquet(str(PARQUET/"order_payments"))
S = {"orders_rows": orders.count(), "payments_rows": pay.count(),
     "orders_parts": orders.rdd.getNumPartitions(), "payments_parts": pay.rdd.getNumPartitions()}
print(S)
# 2. lazy: 3 transformations
sc.setJobGroup("P4_S_LAZY","Самост.: lazy")
pay_prep = (pay.select("order_id","payment_type","payment_installments","payment_value")
            .filter(F.col("payment_value")>0)
            .withColumn("value_per_installment", F.col("payment_value")/F.greatest(F.col("payment_installments"),F.lit(1))))
b1=len([j for j in api("jobs") if j.get("jobGroup")=="P4_S_LAZY"])
t0=perf_counter(); n_prep=pay_prep.count(); t_c=perf_counter()-t0
b2=len([j for j in api("jobs") if j.get("jobGroup")=="P4_S_LAZY"])
S["lazy"]={"jobs_before":b1,"jobs_after":b2,"rows":n_prep,"count_s":t_c}; print(S["lazy"])
''')
code('''
# 3. repartition / coalesce
base16 = pay.repartition(16, "payment_type")
rp4, co4 = base16.repartition(4), base16.coalesce(4)
S["plans"] = {"repartition(4)": "RoundRobinPartitioning(4)" in plan_text(rp4), "coalesce(4)": "Coalesce" in plan_text(co4)}
S["prof"] = {}
for name, df, g in [("repartition(4)", rp4, "P4_S_REP"), ("coalesce(4)", co4, "P4_S_COAL")]:
    sc.setJobGroup(g, name); cnt=[r["count"] for r in partition_profile(df).collect()]
    S["prof"][name] = {"sizes":cnt,"min":min(cnt),"max":max(cnt),"mean":sum(cnt)/len(cnt), **ui_summary(g)}
print(S["plans"]); print(S["prof"])
# 4. groupBy
sc.setJobGroup("P4_S_GB","Самост.: groupBy payment_type")
gb = pay.groupBy("payment_type").agg(F.count("*").alias("n"), F.round(F.sum("payment_value"),2).alias("sum_value")).orderBy(F.desc("sum_value"))
gb.show(); S["groupby_rows"]=[tuple(r) for r in gb.collect()]
S["groupby_exchange"] = "hashpartitioning(payment_type" in plan_text(gb)
S["groupby_ui"]=ui_summary("P4_S_GB"); print(S["groupby_ui"])
''')
code('''
# 5. SMJ vs BHJ (orders x order_payments по order_id)
sc.setJobGroup("P4_S_SMJ","Самост.: SMJ")
s_smj = orders.hint("merge").join(pay.hint("merge"), "order_id", "inner")
t0=perf_counter(); n1=s_smj.count(); ts1=perf_counter()-t0
sc.setJobGroup("P4_S_BHJ","Самост.: BHJ")
s_bhj = orders.join(F.broadcast(pay), "order_id", "inner")
t0=perf_counter(); n2=s_bhj.count(); ts2=perf_counter()-t0
p1,p2 = plan_text(s_smj), plan_text(s_bhj)
S["join"]={"SMJ":{"rows":n1,"time":ts1,"exchange":p1.count(") Exchange"),"ok":"SortMergeJoin" in p1,**ui_summary("P4_S_SMJ")},
           "BHJ":{"rows":n2,"time":ts2,"exchange":p2.count(") Exchange"),"ok":"BroadcastHashJoin" in p2,**ui_summary("P4_S_BHJ")}}
print(json.dumps(S["join"],ensure_ascii=False,indent=1))
# 6. persist
spark.catalog.clearCache()
j = (orders.select("order_id","customer_id","order_status").join(F.broadcast(pay), "order_id").withColumn("v2", F.col("payment_value")*2))
def a1(df): return df.groupBy("payment_type").agg(F.sum("v2").alias("s")).agg(F.sum("s")).first()[0]
def a2(df): return df.groupBy("order_status").agg(F.sum("v2").alias("s")).agg(F.sum("s")).first()[0]
_ = a1(j)
t0=perf_counter(); a1(j); t1=perf_counter(); a2(j); t2=perf_counter()
cj = j.persist(StorageLevel.MEMORY_AND_DISK)
cj.count(); t3=perf_counter(); a1(cj); t4=perf_counter(); a2(cj); t5=perf_counter()
S["cache"]={"nc1":t1-t0,"nc2":t2-t1,"mat":t3-t2,"c1":t4-t3,"c2":t5-t4}; cj.unpersist(blocking=True); print(S["cache"])
''')
code('''
# 7. серия partitions (4 значения, 3 повтора, медиана)
def bench_join(p, repeats=3):
    ts=[]
    for _ in range(repeats):
        a = orders.repartition(p); b = pay.repartition(p)
        t0=perf_counter(); a.join(b,"order_id").groupBy("payment_type").count().collect(); ts.append(perf_counter()-t0)
    return ts
_ = bench_join(4,1)
S["bench"]=[]
for p in [1,4,16,64]:
    ts=bench_join(p); S["bench"].append({"p":p,"runs":ts,"median":median(ts)}); print(p,[round(t,3) for t in ts])
for b in S["bench"]: b["speedup"]=S["bench"][0]["median"]/b["median"]
# 8. запись с двумя p
S["small"]=[]
for p in [2,32]:
    r=write_and_measure(pay,"pay",p); S["small"].append({"p":p,"files":r[0],"total_mib":r[1],"avg_kib":r[2]}); print(p,r)
R["self"]=S
''')

md("## Итоги и завершение сессии")
code('''
R["ui_url"] = sc.uiWebUrl
# Итоговые Jobs/Stages из Spark UI (REST API) вместо скриншотов вкладки Jobs
R["ui_jobs"] = [{"group": j.get("jobGroup"), "jobId": j["jobId"], "name": j["name"][:70], "stages": len(j["stageIds"]), "tasks": j["numTasks"], "status": j["status"]} for j in api("jobs")]
print("Всего Jobs:", len(R["ui_jobs"]))
def default(o):
    try: return float(o)
    except Exception: return str(o)
with open("results.json","w",encoding="utf-8") as f: json.dump(R, f, ensure_ascii=False, indent=1, default=default)
spark.stop()
''')
nb.cells = C
nb.metadata["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
nbf.write(nb, "Практика_4.ipynb")
