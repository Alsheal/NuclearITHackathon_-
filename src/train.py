"""Обучение: scaffold-CV -> финальная модель -> models/si_model.joblib.
Запуск: python -m src.train [--no-external]"""
import sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from src.features import build, scaffold
from src.model import make_model

ROOT = Path(__file__).resolve().parents[1]


SI_CLIP = 1e5          # SI внешних данных выше этого значения считаем выбросом (ошибка единиц)
DROP_TEST_OVERLAP = True


def canon(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m else None


def load_training_data(use_external=True):
    """Официальный train.csv + внешние данные data/external/*.csv (колонки SMILES, IC50_uM, CC50_uM)."""
    tr = pd.read_csv(ROOT / "data/raw/train.csv")
    # NaN в CC50 ничего не означает -> строки убираем. NaN в IC50 = очень малый IC50, SI = 10000 -> оставляем.
    tr = tr.dropna(subset=["SMILES", "SI VACV"]).reset_index(drop=True)
    tr["is_competition"] = True
    parts = [tr]
    ext_files = sorted((ROOT / "data/external").glob("*.csv")) if use_external else []
    if ext_files:
        known = set(tr.SMILES.map(canon))
        if DROP_TEST_OVERLAP:
            known |= set(pd.read_csv(ROOT / "data/raw/test.csv").SMILES.map(canon))
    for f in ext_files:
        e = pd.read_csv(f)
        if not {"SMILES", "IC50_uM", "CC50_uM"} <= set(e.columns):
            continue                                       # не наш формат (например, data_report)
        n0 = len(e)
        e = e.dropna(subset=["SMILES", "IC50_uM", "CC50_uM"])
        e = e[(e.IC50_uM > 0) & (e.CC50_uM > 0)]
        e = e[~e.SMILES.map(canon).isin(known)]            # дубли train и (опц.) пересечение с test
        e["IC50_uM_VACV"] = e["IC50_uM"]
        e["SI VACV"] = (e["CC50_uM"] / e["IC50_uM"]).clip(upper=SI_CLIP)   # SI пересчитан по формуле
        e["is_competition"] = False
        parts.append(e[["SMILES", "CC50_uM", "IC50_uM_VACV", "SI VACV", "is_competition"]])
        print(f"External: {f.name}: {n0} -> {len(e)} rows used")
    return pd.concat(parts, ignore_index=True)


def main():
    tr = load_training_data(use_external="--no-external" not in sys.argv)
    comp = tr["is_competition"].to_numpy()               # метрики считаем только на данных соревнования
    X = build(tr.SMILES)
    y = np.log10(1 + tr["SI VACV"].to_numpy(float))      # все строки, включая SI=0
    groups = tr.SMILES.map(scaffold).to_numpy()

    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        oof[b] = make_model().fit(X[a], y[a]).predict(X[b])
    si, pred = tr["SI VACV"].to_numpy(float)[comp], np.clip(10 ** oof - 1, 0, None)[comp]
    oof, y_c = oof[comp], y[comp]
    metrics = {
        "mode": "SMILES only (descriptors+MACCS+Morgan)",
        "train_rows_total": int(len(tr)), "eval_rows_competition": int(comp.sum()),
        "RMSE_log10(1+SI)": float(np.sqrt(np.mean((oof - y_c) ** 2))),
        "MAE_log10(1+SI)": float(np.mean(np.abs(oof - y_c))),
        "MAE_SI": float(np.mean(np.abs(pred - si))),
        "Spearman": float(pd.Series(oof).corr(pd.Series(y_c), method="spearman")),
    }
    print("Scaffold-CV:", metrics)
    pd.Series(metrics).to_json(ROOT / "results/cv_metrics.json", indent=2)

    model = make_model().fit(X, y)
    joblib.dump(model, ROOT / "models/si_model.joblib")
    print("Saved models/si_model.joblib")


if __name__ == "__main__":
    main()
