"""Предсказание -> results/submission.csv.  Запуск: python -m src.predict"""
from pathlib import Path
import joblib, numpy as np, pandas as pd
from src.features import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = joblib.load(ROOT / "models/si_model.joblib")
    te = pd.read_csv(ROOT / "data/raw/test.csv")

    pred = np.clip(
        10 ** model.predict(build(te["SMILES"])) - 1,
        0,
        None
    )

    sub = pd.DataFrame({
        "index": te["index"],
        "SI VACV": pred
    })

    sample = pd.read_csv(ROOT / "data/raw/sample_submit.csv")

    assert len(sub) == len(sample), "Неверное количество строк"
    assert sub["index"].equals(sample["index"]), "Неверный порядок index"
    assert np.isfinite(sub["SI VACV"]).all(), "Есть бесконечные или NaN значения"
    assert (sub["SI VACV"] >= 0).all(), "Есть отрицательные предсказания"

    sub.to_csv(ROOT / "results/submission.csv", index=False)

    print("Saved results/submission.csv")
    print(sub["SI VACV"].describe())


if __name__ == "__main__":
    main()