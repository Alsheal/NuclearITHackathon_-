import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    rdFingerprintGenerator,
    Descriptors,
    Lipinski,
    Crippen,
    rdMolDescriptors
)

FP_SIZE = 2048
FP_RADIUS = 2

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_SIZE
)


def smiles_to_features(smiles_list):

    features = []

    for smiles in smiles_list:

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Morgan Fingerprint
        fp = generator.GetFingerprint(mol)

        fp_array = np.zeros(
            FP_SIZE,
            dtype=np.float32
        )

        DataStructs.ConvertToNumpyArray(
            fp,
            fp_array
        )

        # Молекулярные характеристики
        descriptors = np.array([
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Lipinski.HeavyAtomCount(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Lipinski.NumHeteroatoms(mol)
        ], dtype=np.float32)

        molecule = np.concatenate([
            fp_array,
            descriptors
        ])

        features.append(molecule)

    return np.array(features, dtype=np.float32)