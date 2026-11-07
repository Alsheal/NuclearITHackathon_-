"""Обучение: scaffold-CV -> финальная модель -> models/si_model.joblib.
Запуск из корня: python -m src.train [--no-external]
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.features import build, scaffold
from src.model import make_model

ROOT = Path(__file__).resolve().parents[1]
SI_CLIP = 5000.0
DROP_TEST_OVERLAP = True


def canon(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m else None


def load_training_data(use_external=True):
    """Официальный train.csv + внешние данные data/external/*.csv"""
    from src.data_io import read_messy_csv   # уже есть в проекте

    # Читаем основной train более устойчиво
    try:
        tr = pd.read_csv(ROOT / "data/raw/train.csv")
    except Exception:
        tr = read_messy_csv(ROOT / "data/raw/train.csv")

    tr = tr.dropna(subset=["SMILES", "SI VACV"]).reset_index(drop=True)
    tr["is_competition"] = True
    parts = [tr]

    ext_files = sorted((ROOT / "data/external").glob("*.csv")) if use_external else []
    if ext_files:
        known = set(tr.SMILES.map(canon))
        if DROP_TEST_OVERLAP:
            try:
                test = pd.read_csv(ROOT / "data/raw/test.csv")
            except Exception:
                test = read_messy_csv(ROOT / "data/raw/test.csv")
            known |= set(test.SMILES.map(canon))

    for f in ext_files:
        try:
            e = pd.read_csv(f)
        except Exception:
            e = read_messy_csv(f)

        if not {"SMILES", "IC50_uM", "CC50_uM"} <= set(e.columns):
            print(f"Skip {f.name}: нет нужных колонок")
            continue

        n0 = len(e)
        e = e.dropna(subset=["SMILES", "IC50_uM", "CC50_uM"])
        e = e[(e.IC50_uM > 0) & (e.CC50_uM > 0)]
        e = e[~e.SMILES.map(canon).isin(known)]
        e["IC50_uM_VACV"] = e["IC50_uM"]
        e["SI VACV"] = (e["CC50_uM"] / e["IC50_uM"]).clip(upper=SI_CLIP)
        e["is_competition"] = False
        parts.append(e[["SMILES", "CC50_uM", "IC50_uM_VACV", "SI VACV", "is_competition"]])
        print(f"External: {f.name}: {n0} -> {len(e)} rows used")

    return pd.concat(parts, ignore_index=True)

def main():
    tr = load_training_data(use_external="--no-external" not in sys.argv)
    comp = tr["is_competition"].to_numpy()
    X = build(tr["SMILES"])
    y = np.log10(1.0 + tr["SI VACV"].to_numpy(float))
    groups = tr["SMILES"].map(scaffold).to_numpy()

    # Веса: конкурсные строки важнее
    weights = np.where(comp, 2.5, 1.0)

    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        m = make_model()
        m.fit(X[a], y[a], sample_weight=weights[a])
        oof[b] = m.predict(X[b])

    si_true = tr["SI VACV"].to_numpy(float)[comp]
    pred = np.clip(10 ** oof[comp] - 1.0, 0, None)
    oof_c, y_c = oof[comp], y[comp]

    metrics = {
        "mode": "SMILES only (descriptors+MACCS+Morgan2048)",
        "train_rows_total": int(len(tr)),
        "eval_rows_competition": int(comp.sum()),
        "RMSE_log10(1+SI)": float(np.sqrt(np.mean((oof_c - y_c) ** 2))),
        "MAE_log10(1+SI)": float(np.mean(np.abs(oof_c - y_c))),
        "MAE_SI": float(np.mean(np.abs(pred - si_true))),
        "Spearman": float(pd.Series(oof_c).corr(pd.Series(y_c), method="spearman")),
    }
    print("Scaffold-CV:", metrics)
    (ROOT / "results").mkdir(exist_ok=True)
    pd.Series(metrics).to_json(ROOT / "results/cv_metrics.json", indent=2)

    model = make_model()
    model.fit(X, y, sample_weight=weights)
    joblib.dump({"model": model, "si_pred_clip": 5000.0, "si_scale": 1.0},
                ROOT / "models/si_model.joblib")
    print("Saved models/si_model.joblib")


if __name__ == "__main__":
    main()