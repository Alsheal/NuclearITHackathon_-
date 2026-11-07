import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import Draw

from src import demo_backend as B


def draw_mol(mol, size=(420, 320)):
    return Draw.MolToImage(mol, size=size)


def hist_si(values, title):
    """Гистограмма log10(SI) с линией порога SI = 10 (нули и мелочь отсечены на 0.01)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(np.log10(np.clip(v, 1e-2, None)), bins=25)
    ax.axvline(np.log10(B.SI_THRESHOLD), ls="--", c="r", label=f"SI = {B.SI_THRESHOLD:g}")
    ax.set(xlabel="log10(SI)", ylabel="Молекул", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def top_plot(df, n=15):
    top = df.dropna(subset=["SI"]).nlargest(n, "SI").iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(top) + 1))
    ax.barh([s[:28] + ("…" if len(s) > 28 else "") for s in top["smiles"]], top["SI"])
    ax.axvline(B.SI_THRESHOLD, ls="--", c="r")
    ax.set_xlabel("Прогноз SI")
    fig.tight_layout()
    return fig


def importance_plot(model):
    """Важность признаков: дескрипторы по отдельности, отпечаток Моргана одной суммой."""
    imp = getattr(B.forest_of(model), "feature_importances_", None)
    if imp is None:
        return None
    n_tail = len(imp) - B.FP_BITS
    if n_tail == len(B.DESCRIPTOR_NAMES) + 2:
        tail_names = B.DESCRIPTOR_NAMES + B.CC50_NAMES
    elif n_tail == len(B.DESCRIPTOR_NAMES):
        tail_names = B.DESCRIPTOR_NAMES
    else:
        return None
    names = ["Morgan fingerprint (сумма)"] + tail_names
    vals = np.concatenate([[imp[:B.FP_BITS].sum()], imp[B.FP_BITS:]])
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.barh([names[i] for i in order], vals[order])
    ax.set_xlabel("Важность признака")
    fig.tight_layout()
    return fig