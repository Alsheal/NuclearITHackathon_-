import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator


# Параметры обработки молекул
RADIUS = 2
FP_SIZE = 2048


def smiles_to_features(smiles_list):
    """
    Преобразует список SMILES в числовые признаки.

    Используются:
    1. Morgan Fingerprints
    2. Молекулярная масса
    3. LogP
    4. TPSA
    5. Количество колец
    6. Количество доноров водородных связей
    7. Количество акцепторов водородных связей
    """

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE
    )

    features = []

    for smiles in smiles_list:

        molecule = Chem.MolFromSmiles(str(smiles))

        if molecule is None:
            raise ValueError(
                f"Некорректная SMILES-запись: {smiles}"
            )

        # Morgan Fingerprint
        fingerprint = generator.GetFingerprint(molecule)

        fingerprint_array = np.zeros(
            (FP_SIZE,),
            dtype=np.float32
        )

        for index in fingerprint.GetOnBits():
            fingerprint_array[index] = 1.0

        # Дополнительные химические характеристики
        descriptors = np.array([
            Descriptors.MolWt(molecule),
            Descriptors.MolLogP(molecule),
            Descriptors.TPSA(molecule),
            Descriptors.RingCount(molecule),
            Descriptors.NumHDonors(molecule),
            Descriptors.NumHAcceptors(molecule)
        ], dtype=np.float32)

        # Объединяем все признаки
        molecule_features = np.concatenate([
            fingerprint_array,
            descriptors
        ])

        features.append(molecule_features)

    return np.array(features, dtype=np.float32)


def predict_si(smiles_list, model_package):
    """
    Предсказывает индекс селективности SI
    для переданных молекул.
    """

    features = smiles_to_features(smiles_list)

    model = model_package["model"]

    predicted_log_si = model.predict(features)

    # Возвращаемся от log10(SI) к обычному SI
    predicted_si = np.power(10, predicted_log_si)

    return predicted_si


def predict_si_log(smiles_list, model_package):
    """
    Предсказывает log10(SI).
    """

    features = smiles_to_features(smiles_list)

    model = model_package["model"]

    return model.predict(features)