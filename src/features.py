
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


FINGERPRINT_BITS = 2048
MORGAN_RADIUS = 2


def is_valid_smiles(smiles):
    """Проверяет корректность SMILES."""

    if not isinstance(smiles, str) or not smiles.strip():
        return False

    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def smiles_to_features(smiles, cc50_values=None):
    """Преобразует SMILES в числовые признаки."""

    smiles = list(smiles)

    if cc50_values is None:
        cc50_values = [np.nan] * len(smiles)
    else:
        cc50_values = list(cc50_values)

    if len(smiles) != len(cc50_values):
        raise ValueError("Количество SMILES и CC50 не совпадает.")

    result = []

    for i, (smile, cc50) in enumerate(zip(smiles, cc50_values)):

        if not is_valid_smiles(smile):
            raise ValueError(
                f"Некорректный SMILES в строке {i}: {smile}"
            )

        mol = Chem.MolFromSmiles(smile)

        # Morgan fingerprints — описание структуры молекулы
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            MORGAN_RADIUS,
            nBits=FINGERPRINT_BITS
        )

        fingerprint_array = np.zeros(
            FINGERPRINT_BITS,
            dtype=np.float32
        )

        DataStructs.ConvertToNumpyArray(
            fingerprint,
            fingerprint_array
        )

        # Физико-химические характеристики
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol)
        ]

        # Обработка CC50
        try:
            cc50 = float(cc50)
        except (TypeError, ValueError):
            cc50 = np.nan

        missing_cc50 = (
            not np.isfinite(cc50) or cc50 <= 0
        )

        if missing_cc50:
            log_cc50 = np.nan
        else:
            log_cc50 = np.log10(cc50)

        features = np.concatenate([
            fingerprint_array,
            np.array(descriptors, dtype=np.float32),
            np.array(
                [log_cc50, float(missing_cc50)],
                dtype=np.float32
            )
        ])

        result.append(features)

    return np.array(result, dtype=np.float32)