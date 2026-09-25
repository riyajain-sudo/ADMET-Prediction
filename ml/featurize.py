"""
Step 3: The ONE place molecule -> feature-vector conversion happens.

This module is imported by both:
  - training scripts (Step 4)       -> featurize thousands of molecules at once
  - the FastAPI /predict endpoint (Step 6) -> featurize one user-submitted SMILES

That symmetry is the whole point. If training and inference ever computed
features differently -- even a subtly different fingerprint radius or bit
count -- the model would be scored on inputs unlike anything it saw during
training ("train/serve skew"), which fails silently: no error, just quietly
wrong predictions. Keeping exactly one function makes that class of bug
structurally impossible.

Technique: Morgan fingerprints (aka ECFP), the standard choice used in the
published MoleculeNet Random Forest baselines we benchmark against in Step 4.
For each atom, RDKit looks at its neighborhood out to FINGERPRINT_RADIUS bonds
and hashes that local substructure into one of FINGERPRINT_N_BITS bit
positions. A molecule becomes a fixed-length 0/1 vector regardless of how many
atoms it actually has -- which is what makes it usable as input to a
scikit-learn model.

We also capture "bit info": for every bit that gets set to 1, which atom(s)
in the molecule caused it. This has no effect on the model or its
predictions -- it's inert extra output, computed for free alongside the
fingerprint. It only gets used in Step 5, where SHAP explains which bits
mattered most and we trace those bits back to real atoms to highlight.
"""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# RDKit's C++ layer logs parse errors/warnings straight to stderr by default.
# We handle invalid SMILES explicitly (return None), so those messages would
# just be noise -- especially once this runs inside the API, where every bad
# user input would otherwise spam the server log with RDKit internals.
RDLogger.DisableLog("rdApp.*")

# Standard "ECFP4" settings (radius 2 = 4-bond diameter) used by the
# published MoleculeNet RF baselines. Changing these would break the
# benchmark comparison in Step 4, so they're pinned here as constants.
FINGERPRINT_RADIUS = 2
FINGERPRINT_N_BITS = 2048

# Built once at import time and reused for every call -- RDKit recommends
# this over constructing a new generator per molecule.
_FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_N_BITS
)


@dataclass
class FeaturizedMolecule:
    """Everything downstream steps need from one molecule, computed once."""

    smiles: str
    mol: Chem.Mol
    fingerprint: np.ndarray  # shape (FINGERPRINT_N_BITS,), dtype uint8 -- model input
    bit_info: dict  # bit_index -> ((atom_index, radius), ...) -- used by Step 5 only


def featurize_smiles(smiles: str) -> FeaturizedMolecule | None:
    """
    Convert one SMILES string into a FeaturizedMolecule.

    Returns None if the SMILES can't be parsed by RDKit at all (invalid
    molecule). Step 2 found this happens for a small fraction of BBBP and
    Tox21 -- callers must handle None explicitly rather than assuming every
    input featurizes successfully.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_info: dict = {}
    additional_output = rdFingerprintGenerator.AdditionalOutput()
    additional_output.AllocateBitInfoMap()

    fp = _FINGERPRINT_GENERATOR.GetFingerprint(mol, additionalOutput=additional_output)

    fingerprint = np.zeros((FINGERPRINT_N_BITS,), dtype=np.uint8)
    for bit in fp.GetOnBits():
        fingerprint[bit] = 1

    bit_info = dict(additional_output.GetBitInfoMap())

    return FeaturizedMolecule(
        smiles=smiles,
        mol=mol,
        fingerprint=fingerprint,
        bit_info=bit_info,
    )


def featurize_batch(smiles_list: list[str]) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Featurize many SMILES at once, for training (Step 4).

    Built on top of featurize_smiles -- no separate fingerprint logic here,
    just looping and stacking, so training can never drift from the
    single-molecule path used at inference time.

    Returns:
        X: (n_valid, FINGERPRINT_N_BITS) array, one row per VALID molecule
        valid_indices: original list positions that were valid, in order --
            use this to select the matching rows from your target/label array
        invalid_indices: original list positions that failed to parse
    """
    fingerprints = []
    valid_indices = []
    invalid_indices = []

    for i, smiles in enumerate(smiles_list):
        result = featurize_smiles(smiles)
        if result is None:
            invalid_indices.append(i)
            continue
        fingerprints.append(result.fingerprint)
        valid_indices.append(i)

    X = np.stack(fingerprints) if fingerprints else np.empty((0, FINGERPRINT_N_BITS), dtype=np.uint8)
    return X, valid_indices, invalid_indices
