from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.features import build

ROOT = Path(__file__).resolve().parents[1]


def main():
    obj = joblib.load(ROOT / "models/si_model.joblib")
    if isinstance(obj, dict):
        model = obj["model"]
        clip = float(obj.get("si_pred_clip", 5000.0))
        scale = float(obj.get("si_scale", 1.0))
    else:
        model, clip, scale = obj, 5000.0, 1.0

    te = pd.read_csv(ROOT / "data/raw/test.csv")
    raw = model.predict(build(te["SMILES"]))
    pred = np.clip(scale * (10.0 ** raw - 1.0), 0.0, clip)

    sub = pd.DataFrame({"index": te["index"], "SI VACV": pred})
    sample = pd.read_csv(ROOT / "data/raw/sample_submit.csv")
    assert len(sub) == len(sample)
    assert sub["index"].equals(sample["index"])

    (ROOT / "results").mkdir(exist_ok=True)
    sub.to_csv(ROOT / "results/submission.csv", index=False)
    print("Saved results/submission.csv")
    print(sub["SI VACV"].describe())


if __name__ == "__main__":
    main()