"""
Step 4: train/test splitting -- training-only, unlike featurize.py.

This is NOT shared with inference. Splitting only makes sense when you have
a whole dataset to divide; the live API only ever sees one molecule per
request, so there's nothing here for it to reuse.

Two strategies, matching what the original MoleculeNet paper (Wu et al.
2018) used per dataset -- using a different split than the paper would make
the Step 4 benchmark comparison meaningless:

  - scaffold_split: used for BBBP. Groups molecules by their Bemis-Murcko
    scaffold (the core ring system + linkers, ignoring side chains/R-groups)
    and keeps each scaffold group entirely on one side of the split. This is
    the harder, more realistic test: a model can't pass just by recognizing
    near-duplicate molecules it already saw in training.

  - random_split: used for ESOL and Tox21. Ordinary shuffled split.
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold_for_smiles(smiles: str) -> str:
    """Bemis-Murcko scaffold as a SMILES string; '' if it can't be computed
    (treated as its own group, so such molecules still land somewhere)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(
    smiles_list: list[str], frac_train: float = 0.8, seed: int = 42
) -> tuple[list[int], list[int]]:
    """
    Split molecule indices by scaffold group so no scaffold appears on both
    sides. Deterministic given `seed`: scaffold groups are shuffled before
    sorting by size so tie-breaking is reproducible, not dependent on
    dict/hash ordering.

    Largest scaffold groups are assigned to train first (this matches the
    common "greedy largest-first" scaffold splitter used elsewhere in the
    field) so train ends up with broad structural coverage while test still
    gets a real mix of scaffold sizes, not just singletons.
    """
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        scaffold_to_indices[_scaffold_for_smiles(smiles)].append(i)

    groups = list(scaffold_to_indices.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_train_target = int(frac_train * len(smiles_list))

    train_indices: list[int] = []
    test_indices: list[int] = []
    for group in groups:
        if len(train_indices) + len(group) <= n_train_target:
            train_indices.extend(group)
        else:
            test_indices.extend(group)

    return train_indices, test_indices


def random_split(n: int, frac_train: float = 0.8, seed: int = 42) -> tuple[list[int], list[int]]:
    """Ordinary shuffled split over n items, used for ESOL and Tox21."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(frac_train * n)
    return indices[:n_train].tolist(), indices[n_train:].tolist()
