"""Мост между демо и моделью участника №2.
Если модель не найдена, работает заглушка (MOCK), чтобы интерфейс можно было показывать уже сейчас.
Поддерживает оба формата пакета: новый (models/best_model.pkl, цель = сырой SI)
и старый (models/si_model.joblib, цель = log10(SI)).
"""
import os

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

MODEL_PATHS = ["models/best_model.pkl", "models/si_model.joblib"]
COMPARISON_PATH = "results/model_comparison.csv"
TEST_PATH = "results/test_predictions.csv"  # необязательный: колонки SI_true, SI_predicted
SI_THRESHOLD = 10.0  # общепринятый порог "перспективности", не из условий задачи
# Порядок совпадает с src/features.py участника №2 (последние 10 признаков вектора)
DESCRIPTOR_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds",
                    "RingCount", "HeavyAtoms", "FractionCSP3", "Heteroatoms"]


def load_package():
    """Возвращает пакет модели или None, если модель ещё не обучена."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path)
    return None


def load_comparison():
    return pd.read_csv(COMPARISON_PATH) if os.path.exists(COMPARISON_PATH) else None


def load_test_predictions():
    if os.path.exists(TEST_PATH):
        df = pd.read_csv(TEST_PATH)
        if {"SI_true", "SI_predicted"} <= set(df.columns):
            return df
    return None


def _featurize(smiles_list):
    try:
        from src.features import smiles_to_features
    except ImportError:  # старая версия кода участника №2
        from src.si_model import smiles_to_features
    return smiles_to_features(smiles_list)


def _to_si(raw, package):
    """Выход модели -> SI. Учитывает, учили ли модель на log10(SI) или на сыром SI."""
    if "log10" in str(package.get("target", "")).lower():
        return float(10 ** raw)
    return float(max(raw, 0.0))  # регрессия на сыром SI может дать отрицательное число


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


def _mock_si(smiles):
    import hashlib
    h = int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16)
    return 10 ** (-0.5 + 3.0 * (h % 1000) / 1000)


def predict(smiles, package=None):
    """Единая точка входа для демо: SMILES -> dict."""
    smiles = (smiles or "").strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {"valid": False, "error": "Некорректная или пустая SMILES-запись"}

    std = None
    if package is None:
        si, is_mock = _mock_si(smiles), True
    else:
        X = _featurize([smiles])
        model = package["model"]
        si, is_mock = _to_si(float(model.predict(X)[0]), package), False
        if hasattr(model, "estimators_"):  # разброс деревьев как грубая оценка уверенности
            std = float(np.std([_to_si(float(t.predict(X)[0]), package) for t in model.estimators_]))

    return {
        "valid": True,
        "smiles": smiles,
        "mol": mol,
        "SI": si,
        "log10_SI": float(np.log10(max(si, 1e-2))),
        "std_SI": std,
        "active": si >= SI_THRESHOLD,
        "descriptors": describe(mol),
        "is_mock": is_mock,
    }


def predict_many(smiles_list, package=None):
    rows = []
    for s in smiles_list:
        r = predict(str(s), package)
        rows.append({
            "smiles": s,
            "SI": round(r["SI"], 2) if r["valid"] else None,
            "log10_SI": round(r["log10_SI"], 3) if r["valid"] else None,
            "active_SI>=10": r["active"] if r["valid"] else None,
            "error": r.get("error", ""),
        })
    return pd.DataFrame(rows)