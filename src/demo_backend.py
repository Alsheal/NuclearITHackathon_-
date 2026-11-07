"""Мост между демо и результатами остальных участников.
Демо ничего не обучает и не собирает: оно читает готовые файлы и показывает их.
Если модели нет, работает заглушка (MOCK).
"""
import functools
import json
import os

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

MODEL_PATH = "models/si_model.joblib"        # train.py (участник №2)
REPORT_PATH = "results/training_report.json"  # train.py
SUBMISSION_PATH = "results/submission.csv"    # predict.py
TRAIN_PATH = "data/train.csv"                 # данные организаторов
EXPERIMENTS_PATH = "experiments/results.csv"  # таблица экспериментов
EXTERNAL_CANDIDATES = ["data/external_raw.csv", "data/final_dataset.csv",
                       "data/train_chembl.csv", "train_chembl.csv"]  # участники №1 и №3
SI_THRESHOLD = 10.0  # общепринятый порог "перспективности", не из условий задачи
# Хвост вектора признаков из src/features.py: 9 дескрипторов (+ log10 CC50 и флаг пропуска)
DESCRIPTOR_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "Rings", "RotBonds",
                    "FractionCSP3", "HeavyAtoms"]
CC50_NAMES = ["log10 CC50", "CC50 отсутствует"]
FP_BITS = 2048


def read_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def read_external():
    for p in EXTERNAL_CANDIDATES:
        if os.path.exists(p):
            return p, pd.read_csv(p)
    return None, None


def read_submission():
    df = read_csv(SUBMISSION_PATH)
    if df is None:
        return None
    col = "SI VACV" if "SI VACV" in df.columns else df.columns[-1]
    return df[col]


def load_model():
    """-> (модель, текст ошибки). Модель: sklearn Pipeline (импутер + Random Forest)."""
    if not os.path.exists(MODEL_PATH):
        return None, f"Файл {MODEL_PATH} не найден"
    try:
        obj = joblib.load(MODEL_PATH)
        return (obj["model"] if isinstance(obj, dict) else obj), None  # поддержка старого формата
    except Exception as e:  # например, другая версия scikit-learn
        return None, f"Не удалось загрузить модель: {e}"


def forest_of(model):
    return list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model


@functools.lru_cache(maxsize=4)
def _log10_flag(mtime):
    rep = read_report() or {}
    return "log10" in str(rep.get("target", "log10(SI)")).lower()


def _is_log10():
    """Учили ли модель на log10(SI). Перечитывается, если отчёт изменился."""
    return _log10_flag(os.path.getmtime(REPORT_PATH) if os.path.exists(REPORT_PATH) else 0)


def _to_si(raw):
    return float(10 ** raw) if _is_log10() else float(max(raw, 0.0))


def _featurize(smiles_list):
    from src.features import smiles_to_features
    return smiles_to_features(smiles_list)  # CC50 в демо не передаём: есть только SMILES


def _mock_si(smiles):
    import hashlib
    h = int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16)
    return 10 ** (-0.5 + 3.0 * (h % 1000) / 1000)


def describe(mol):
    return {
        "Молекулярная формула": rdMolDescriptors.CalcMolFormula(mol),
        "Молекулярная масса": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "Атомов (без H)": mol.GetNumAtoms(),
        "Связей": mol.GetNumBonds(),
        "Колец": rdMolDescriptors.CalcNumRings(mol),
        "Доноров H-связи": rdMolDescriptors.CalcNumHBD(mol),
        "Акцепторов H-связи": rdMolDescriptors.CalcNumHBA(mol),
    }


def lipinski_notes(mol):
    """Правило пяти Липински: быстрая подсказка о «лекарственности». Не связана с моделью."""
    checks = [("масса > 500", Descriptors.MolWt(mol) > 500),
              ("LogP > 5", Descriptors.MolLogP(mol) > 5),
              ("доноров H-связи > 5", rdMolDescriptors.CalcNumHBD(mol) > 5),
              ("акцепторов H-связи > 10", rdMolDescriptors.CalcNumHBA(mol) > 10)]
    return [name for name, bad in checks if bad]


def predict(smiles, model=None):
    """Одна молекула: SMILES -> dict."""
    smiles = (smiles or "").strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {"valid": False, "error": "Некорректная или пустая SMILES-запись"}

    low = high = None
    if model is None:
        si = _mock_si(smiles)
    else:
        X = _featurize([smiles])
        p = float(model.predict(X)[0])
        si = _to_si(p)
        forest = forest_of(model)
        if hasattr(forest, "estimators_"):  # разброс деревьев = грубая оценка уверенности
            Xt = model[:-1].transform(X) if hasattr(model, "named_steps") else X
            s = float(np.std([t.predict(Xt)[0] for t in forest.estimators_]))
            low, high = _to_si(p - s), _to_si(p + s)
    return {
        "valid": True, "mol": mol, "SI": si,
        "log10_SI": float(np.log10(max(si, 1e-2))),
        "low": low, "high": high, "active": si >= SI_THRESHOLD,
        "descriptors": describe(mol), "lipinski": lipinski_notes(mol),
        "is_mock": model is None,
    }


def predict_many(smiles_list, model=None):
    """Много молекул (быстро, без оценки разброса)."""
    smiles = [str(s).strip() for s in smiles_list]
    ok = [bool(s) and Chem.MolFromSmiles(s) is not None for s in smiles]
    good = [s for s, k in zip(smiles, ok) if k]
    if model is None:
        sis = [_mock_si(s) for s in good]
    elif good:
        sis = [_to_si(float(p)) for p in model.predict(_featurize(good))]
    else:
        sis = []
    it = iter(sis)
    rows = []
    for s, k in zip(smiles, ok):
        si = next(it) if k else None
        rows.append({
            "smiles": s,
            "SI": round(si, 2) if k else None,
            "log10_SI": round(float(np.log10(max(si, 1e-2))), 3) if k else None,
            "SI>=10": bool(si >= SI_THRESHOLD) if k else None,
            "error": "" if k else "некорректный SMILES",
        })
    return pd.DataFrame(rows)


def project_status(model):
    """Что из общей работы уже есть. Нужно для проверки сборки перед защитой."""
    ext_path, ext = read_external()
    return [
        ("Обученная модель", model is not None, MODEL_PATH),
        ("Отчёт об обучении", os.path.exists(REPORT_PATH), REPORT_PATH),
        ("Предсказания для Kaggle", os.path.exists(SUBMISSION_PATH), SUBMISSION_PATH),
        ("Данные организаторов", os.path.exists(TRAIN_PATH), TRAIN_PATH),
        ("Внешний датасет", ext is not None, ext_path or "data/external_raw.csv"),
        ("Таблица экспериментов", os.path.exists(EXPERIMENTS_PATH), EXPERIMENTS_PATH),
    ]