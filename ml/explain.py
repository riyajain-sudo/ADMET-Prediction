"""
Step 5: explain one prediction by attributing it back to atoms, using SHAP.

Reuses Step 3's `bit_info` (which atom set each fingerprint bit) -- no
separate way of reading the molecule is introduced here. Given a trained
model and a featurized molecule, this module answers: which atoms pushed
the prediction up, which pushed it down, and by how much?

Technique: SHAP's TreeExplainer. This is the reason Random Forest was
chosen as the model in Step 4 -- TreeExplainer computes *exact* SHAP values
for tree ensembles by walking the trees directly, instead of the slower,
approximate, sampling-based explainers SHAP needs for other model types.

Important finding from testing this against our actual models: TreeExplainer's
default mode ("tree_path_dependent") fails its own built-in correctness check
on this scikit-learn version -- the contributions it computes do not actually
add up to the real prediction (off by a lot, not rounding error). Passing an
explicit `background` sample of real molecules and using
feature_perturbation="interventional" fixes this and passes the check
cleanly, so that combination is used everywhere below rather than the
TreeExplainer default.

The atom view below is a heuristic allocation of the full fingerprint SHAP
vector. Inactive bits and any active bit without a valid atom mapping are
reported as an explicit residual; they are not silently dropped.

Known simplification: bit environments may collide and may contain repeated
center atoms. Each bit's contribution is divided equally across its unique
mapped center atoms. This heuristic is not an atom-level SHAP decomposition;
the returned residual preserves contribution mass that cannot be mapped.
"""

from dataclasses import dataclass

import numpy as np
import shap

from ml.featurize import FeaturizedMolecule, featurize_batch

# Index into predict_proba(...)[:, 1] / SHAP's per-class output -- the
# "positive" class (BBB-permeable, or toxic) that Step 4's AUROC was
# computed against. Explanations should target the same thing the model
# was scored on.
POSITIVE_CLASS_INDEX = 1

# Number of real molecules used as the reference point ("what does a
# typical molecule look like?") that interventional SHAP measures every
# bit's contribution against. 100 is the commonly recommended size for
# TreeExplainer's interventional mode -- enough to be a stable reference,
# small enough to stay fast (every prediction re-walks the trees once per
# background row).
BACKGROUND_SIZE = 100


@dataclass
class Explanation:
    """Everything Step 6 (the API) and Step 7 (the frontend) need from one explained prediction."""

    prediction: float  # regression value, or probability of the positive class
    base_value: float  # explainer's baseline prediction, before any bit's contribution
    atom_contributions: dict[int, float]  # heuristic share per atom, not atom-level SHAP
    unmapped_contribution: float  # SHAP mass not assigned to atom centers


def sample_background(smiles_list: list[str], n: int = BACKGROUND_SIZE, seed: int = 42) -> np.ndarray:
    """
    Build a background sample from a list of real SMILES (typically a
    model's own training data), for use with make_explainer().
    """
    X, _valid_indices, _invalid_indices = featurize_batch(smiles_list)
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[chosen].astype(np.float64)


def make_explainer(model, background: np.ndarray) -> shap.TreeExplainer:
    """Build a SHAP explainer for one trained model. See module docstring
    for why interventional mode + an explicit background is required here."""
    return shap.TreeExplainer(model, background, feature_perturbation="interventional")


def explain_fingerprint(
    explainer: shap.TreeExplainer, featurized: FeaturizedMolecule, is_classifier: bool
) -> Explanation:
    """
    Explain one already-featurized molecule's prediction from one model.

    `is_classifier` selects the output shape SHAP returns: classifiers
    produce a per-class SHAP value per bit, regressors produce just one.
    """
    X = featurized.fingerprint.reshape(1, -1).astype(np.float64)
    shap_values = explainer.shap_values(X)

    if is_classifier:
        per_bit = shap_values[0, :, POSITIVE_CLASS_INDEX]
        base_value = float(explainer.expected_value[POSITIVE_CLASS_INDEX])
    else:
        per_bit = shap_values[0]
        base_value = float(explainer.expected_value)

    prediction = base_value + float(per_bit.sum())

    atom_contributions: dict[int, float] = {}
    mapped_bit_indices: set[int] = set()
    for bit_index, environments in featurized.bit_info.items():
        unique_atoms = sorted({int(atom_idx) for atom_idx, _radius in environments
                               if 0 <= int(atom_idx) < featurized.mol.GetNumAtoms()})
        if not unique_atoms:
            continue
        mapped_bit_indices.add(int(bit_index))
        share = float(per_bit[bit_index]) / len(unique_atoms)
        for atom_idx in unique_atoms:
            atom_contributions[atom_idx] = atom_contributions.get(atom_idx, 0.0) + share

    unmapped_contribution = float(sum(float(value) for i, value in enumerate(per_bit)
                                      if i not in mapped_bit_indices))
    return Explanation(prediction=prediction, base_value=base_value,
                       atom_contributions=atom_contributions,
                       unmapped_contribution=unmapped_contribution)
