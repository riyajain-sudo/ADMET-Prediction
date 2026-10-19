import joblib
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from ml.explain import explain_fingerprint, make_explainer, sample_background
from ml.featurize import featurize_smiles

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture(scope="module")
def molecule():
    return featurize_smiles(ASPIRIN)


def _check(model_path, csv_path, is_classifier, molecule, task=None):
    model = joblib.load(model_path)
    if task is not None:
        model = model[task]
    background = sample_background(pd.read_csv(csv_path)["smiles"].tolist())
    explanation = explain_fingerprint(make_explainer(model, background), molecule, is_classifier)

    X = molecule.fingerprint.reshape(1, -1)
    direct = model.predict_proba(X)[0, 1] if is_classifier else model.predict(X)[0]
    # SHAP additivity: base value + all contributions must reproduce the real prediction.
    assert explanation.prediction == pytest.approx(direct, abs=1e-6)
    assert set(explanation.atom_contributions) <= set(range(molecule.mol.GetNumAtoms()))


def test_esol_additivity(molecule):
    _check("ml/models/esol_rf.pkl", "data/raw/esol.csv", False, molecule)


def test_bbbp_additivity(molecule):
    _check("ml/models/bbbp_rf.pkl", "data/raw/bbbp.csv", True, molecule)


def test_tox21_additivity(molecule):
    _check("ml/models/tox21_rf.pkl", "data/raw/tox21.csv", True, molecule, task="NR-AR")


def test_atom_allocation_conserves_mass_with_explicit_residual():
    class FakeExplainer:
        expected_value = 0.25

        def shap_values(self, _x):
            return np.array([[0.6, -0.2, 0.3, 0.05]])

    mol = SimpleNamespace(GetNumAtoms=lambda: 3)
    featurized = SimpleNamespace(
        fingerprint=np.array([1, 0, 1, 0], dtype=np.uint8),
        bit_info={0: ((0, 1), (1, 1), (0, 2)), 2: ((2, 1),)},
        mol=mol,
    )
    result = explain_fingerprint(FakeExplainer(), featurized, is_classifier=False)
    assert result.atom_contributions == pytest.approx({0: 0.3, 1: 0.3, 2: 0.3})
    assert result.unmapped_contribution == pytest.approx(-0.15)
    assert (result.base_value + sum(result.atom_contributions.values()
                                    ) + result.unmapped_contribution) == pytest.approx(result.prediction)
    assert set(result.atom_contributions) <= {0, 1, 2}
