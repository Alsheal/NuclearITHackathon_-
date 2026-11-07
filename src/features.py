"""Признаки только из SMILES: дескрипторы RDKit + MACCS + Morgan counts."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator as G
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Берём только стабильные дескрипторы (без редко падающих)
_DESC_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "FractionCSP3", "RingCount", "HeavyAtomCount", "NumHeteroatoms",
    "BalabanJ", "BertzCT", "Chi0v", "Chi1v", "Kappa1", "Kappa2",
    "LabuteASA", "PEOE_VSA1", "SMR_VSA1", "SlogP_VSA1",
    "NumSaturatedRings", "NumAromaticHeterocycles", "NumAliphaticHeterocycles",
]
_desc = [getattr(Descriptors, n) for n in _DESC_NAMES if hasattr(Descriptors, n)]
_morgan = G.GetMorganGenerator(radius=2, fpSize=2048)
N_FEATURES = len(_desc) + 167 + 2048


def mol_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.full(N_FEATURES, np.nan)
    d = []
    for f in _desc:
        try:
            d.append(float(f(mol)))
        except Exception:
            d.append(np.nan)
    d = np.clip(np.nan_to_num(np.array(d, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=float)
    morgan = _morgan.GetCountFingerprintAsNumPy(mol).astype(float)
    return np.concatenate([d, maccs, morgan])


def scaffold(smiles):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(str(smiles)) or str(smiles)
    except Exception:
        return str(smiles)


def build(smiles_series):
    return np.vstack([mol_features(s) for s in smiles_series])