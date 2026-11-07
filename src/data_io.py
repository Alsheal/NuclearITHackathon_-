"""Чтение «грязных» CSV (экспорт из Excel: строки в кавычках, `;` в конце, запятые в Title)."""
import csv
import pandas as pd


def read_messy_csv(path):
    raw = open(path, encoding="utf-8-sig").read().replace("\r", "")
    rows = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.endswith(";"):
            line = line[:-1]
        if line.startswith('"'):
            line = line[1:]
            if line.endswith('"'):
                line = line[:-1]
            line = line.replace('""', '"')
        rows.append(next(csv.reader([line])))
    header, n = rows[0], len(rows[0])
    fixed = []
    for r in rows[1:]:
        if len(r) > n:                              # запятая внутри Title: склеиваем лишние поля
            r = [",".join(r[: len(r) - n + 1])] + r[len(r) - n + 1:]
        if len(r) == n:
            fixed.append(r)
    return pd.DataFrame(fixed, columns=header)
