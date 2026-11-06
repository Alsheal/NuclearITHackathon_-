import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw

from src.demo_backend import DESCRIPTOR_NAMES


def draw_mol(mol, size=(420, 320)):
    return Draw.MolToImage(mol, size=size)


def parity_plot(test_df):
    """Предсказано vs реальное (log10 SI). Нужен results/test_predictions.csv с SI_true, SI_predicted."""
    y = np.log10(np.clip(test_df["SI_true"], 1e-2, None))
    p = np.log10(np.clip(test_df["SI_predicted"], 1e-2, None))
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, p, alpha=0.6)
    ax.plot([lo, hi], [lo, hi], "r--", label="идеал")
    ax.set_xlabel("Реальное log10(SI)")
    ax.set_ylabel("Предсказанное log10(SI)")
    ax.legend()
    fig.tight_layout()
    return fig


def descriptor_importance_plot(package):
    """Важность дескрипторов (последние признаки вектора; биты отпечатка не интерпретируемы).
    Работает для RandomForest/ExtraTrees; у HistGradientBoosting нет feature_importances_."""
    model = package["model"]
    n = len(DESCRIPTOR_NAMES)
    if not hasattr(model, "feature_importances_") or len(model.feature_importances_) < n:
        return None
    imp = model.feature_importances_[-n:]
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh([DESCRIPTOR_NAMES[i] for i in order], imp[order])
    ax.set_xlabel("Важность признака")
    fig.tight_layout()
    return fig