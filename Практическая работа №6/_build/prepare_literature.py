"""Паспорт Literature: чтение ZIP, границы текста, аудит совпадений."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import hashlib
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'practice6_results'
WORK = ROOT / 'practice6_data'
OUT.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)
AUTHORS = ['Chekhov', 'Gogol', 'Turgenev']
EXCLUDE = {
    'Chekhov/Дополнительные вопросы': 'Сатирический опросник, 85 слов: нет полного 300-словного фрагмента.',
    'Chekhov/Вишнёвый сад': 'Пьеса: исключена для сопоставимости повествовательной прозы.',
    'Chekhov/Три сестры': 'Пьеса: исключена для сопоставимости повествовательной прозы.',
    'Chekhov/Встреча весны': 'Фельетон-рассуждение: исключён из основного корпуса рассказов.',
    'Gogol/Ревизор': 'Пьеса.',
    'Gogol/Вечера на хуторе близ Диканьки': 'Сборник; пересекается с отдельной «Ночью перед Рождеством».',
    'Gogol/Ночи на вилле': 'Незавершённый отрывок с редакционным комментарием; связан с «Римом».',
    'Turgenev/Стихотворения в прозе': 'Лирические миниатюры: иной тип текста.',
    'Turgenev/Записки охотника': 'Сборник рассказов; исключён, чтобы единицей было отдельное произведение.'
}
STARTS = {
    'Chekhov/Дом с мезонином': 'Это было шесть-семь лет тому назад',
    'Chekhov/Брак по расчёту': 'В доме вдовы Мымриной',
    'Chekhov/Каштанка': 'Молодая рыжая собака',
    'Chekhov/Палата №6': 'В больничном дворе',
    'Chekhov/Тоска': 'Вечерние сумерки.',
    'Turgenev/Ася': 'Мне было тогда лет двадцать пять',
    'Turgenev/Вешние воды': 'Часу во втором ночи',
    'Turgenev/Дворянское гнездо': 'Весенний, светлый день',
    'Turgenev/Дым': '10\u00a0августа 1862 года',
    'Turgenev/Муму': 'В одной из отдаленных улиц Москвы',
    'Turgenev/Отцы и дети': '–\u00a0Что, Петр, не видать еще?',
    'Turgenev/Первая любовь': 'Гости давно разъехались.',
    'Turgenev/Яков Пасынков': 'Дело было в Петербурге',
    'Gogol/Мёртвые души': 'В ворота гостиницы',
    'Gogol/Нос': 'Марта 25 числа',
    'Gogol/Повесть о том, как поссорился Иван Иванович с Иваном Никифоровичем': 'Славная бекеша',
    'Gogol/Портрет': 'Нигде не останавливалось',
    'Gogol/Рим': 'Попробуй взглянуть',
    'Gogol/Тарас Бульба': '— А поворотись-ка, сын!'
}

def logical_zip_name(info):
    if not info.flag_bits & 0x800:
        try:
            return info.filename.encode('cp437').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return info.filename

def boundaries(text, key):
    if key in STARTS:
        start = text.index(STARTS[key])
    else:
        # Заголовки, авторская строка и римский номер главы находятся
        # перед первой повествовательной строкой; срезы сохраняются в паспорте.
        start = 0
        for line in text.splitlines(keepends=True):
            s = line.strip()
            is_header = (not s or s.isupper() or
                         re.fullmatch(r'[IVXLC]+', s) or
                         s in ['Антон Чехов', 'Николай Гоголь', 'Николай Васильевич Гоголь'] or
                         s == key.split('/')[-1])
            if not is_header:
                break
            start += len(line)
    end = len(text)
    # Примечания в этих изданиях собраны после основного повествования.
    tail = text[start:]
    internal_notes = key in ['Chekhov/Драма на охоте', 'Chekhov/Дуэль', 'Gogol/Тарас Бульба']
    pattern = r'(?m)^\s*Примечания\s*$' if internal_notes else r'(?m)^\s*Примечания\s*$|^\s*1\s*\n'
    candidates = list(re.finditer(pattern, tail))
    if candidates:
        end = start + candidates[0].start()
    if key == 'Chekhov/Драма на охоте':
        end = text.index('\n*\nХорош следователь!')
    # Убираем издательскую датировку в конце.
    date = re.search(r'\n\s*(?:18|19)\d{2}\s*$', text[start:end])
    if date:
        end = start + date.start()
    assert 0 <= start < end <= len(text), key
    return start, end

inventory, selected = [], []
with ZipFile(ROOT / 'Literature.zip') as archive:
    assert sum(i.file_size for i in archive.infolist()) < 250_000_000
    for info in archive.infolist():
        name = logical_zip_name(info)
        path = PurePosixPath(name)
        if path.suffix.lower() != '.txt':
            continue
        assert len(path.parts) == 3 and '..' not in path.parts
        encoding = 'cp1251' if name == 'poems/Pushkin/Руслан и Людмила.txt' else 'utf-8-sig'
        text = archive.read(info).decode(encoding, errors='strict')
        folder, author, filename = path.parts
        key = f'{author}/{path.stem}'
        chosen = folder == 'prose' and author in AUTHORS
        keep = chosen and key not in EXCLUDE
        start, end = boundaries(text, key) if keep else (0, len(text))
        record = dict(document_id=hashlib.sha256(name.encode()).hexdigest(),
                      folder=folder, author=author, filename=filename, encoding=encoding,
                      chars=len(text), work_group=key, body_start=start, body_end=end,
                      keep=keep, reviewed=chosen,
                      decision=EXCLUDE.get(key, 'Границы заголовка и концевых примечаний заданы.' if chosen else 'Вне выбранных авторов/жанра.'))
        body = text[start:end]
        record['body_chars'] = len(body) if keep else 0
        record['body_sha256'] = hashlib.sha256(body.encode()).hexdigest() if keep else ''
        inventory.append(record)
        if keep:
            # Сноски в квадратных скобках, ссылки-номера и заголовки глав
            # удаляются одинаково для всех авторов; пунктуация и словоформы сохранены.
            body = re.sub(r'\[\d+\]', '', body)
            if key in ['Chekhov/Драма на охоте', 'Chekhov/Дуэль', 'Gogol/Тарас Бульба']:
                body = re.sub(r'(?m)^\s*(?:\d+|\*+)\s*\n(?:[^\n]+\n)*(?:\n|$)', '\n', body)
            body = re.sub(r'(?m)^\s*(?:[IVXLC]+|\d+|Глава\s+[^\n]+|Часть\s+[^\n]+|Том\s+[^\n]+)\s*$', '', body)
            selected.append({**record, 'text': body})

manifest = pd.DataFrame(inventory).sort_values(['folder', 'author', 'filename'])
manifest.to_csv(OUT / 'literature_review.csv', index=False)
with (WORK / 'literature_selected.jsonl').open('w', encoding='utf-8') as stream:
    for row in selected:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')

# Аудит не зависит от разбиения: точные хеши и пересечение 50-словных блоков.
# Короткие совпадения общего языка не считаются повторным изданием.
owners, overlaps = {}, []
for row in selected:
    words = re.findall(r'[^\W_]+', row['text'].lower())
    signatures = {hashlib.sha256(' '.join(words[i:i+50]).encode()).hexdigest()
                  for i in range(0, len(words)-49, 25)}
    for signature in signatures:
        if signature in owners and owners[signature] != row['work_group']:
            overlaps.append((owners[signature], row['work_group']))
        else:
            owners[signature] = row['work_group']
assert not pd.DataFrame(selected).body_sha256.duplicated().any()
audit = pd.DataFrame(overlaps, columns=['work_a', 'work_b'])
audit.to_csv(OUT / 'literature_overlap_audit.csv', index=False)
assert not overlaps, 'Обнаружены общие длинные блоки: необходимо уточнить группы.'
preview = '\n\n'.join(f"{x['work_group']} [{x['body_start']}:{x['body_end']}]\nНАЧАЛО {x['text'][:180]!r}\nКОНЕЦ {x['text'][-140:]!r}" for x in selected)
(ROOT / '_build/literature_boundaries.txt').write_text(preview, encoding='utf-8')
print('TXT всего:', len(manifest), '; включено:', len(selected))
print(manifest.query('keep').groupby('author').size().to_string())
print('Точные дубли: 0; общие 50-словные блоки между разными произведениями:', len(overlaps))
