# Технологии и инструментарий анализа больших данных — практические работы

Дицель Н.С., ИКБО-11-23.

## Большие файлы данных

GitHub не принимает файлы больше 100 МБ, поэтому они хранятся в ZIP-архивах (сжатие без потерь).
Перед запуском блокнотов распакуйте архив в ту же папку:

| Архив | Файл | Размер после распаковки |
|---|---|---|
| `Практическая работа №2/ecommerce.sqlite.zip` | `ecommerce.sqlite` | 201 МБ |
| `Практическая работа №3/ecommerce.sqlite.zip` | `ecommerce.sqlite` | 201 МБ |
| `Практическая работа №4/ecommerce.sqlite.zip` | `ecommerce.sqlite` | 201 МБ |
| `Практическая работа №3/transactions.csv.zip` | `transactions.csv` | 271 МБ |

Распаковать все архивы разом (из корня репозитория):

```bash
python -c "import zipfile,glob,os; [zipfile.ZipFile(z).extractall(os.path.dirname(z)) for z in glob.glob('*/*.sqlite.zip')+glob.glob('*/transactions.csv.zip')]"
```

Распакованные файлы перечислены в `.gitignore` и в git не попадают.
