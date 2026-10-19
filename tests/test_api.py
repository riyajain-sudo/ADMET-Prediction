import pytest
from fastapi.testclient import TestClient

from api.main import app

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
NON_CANONICAL_ASPIRIN = "OC(=O)c1ccccc1OC(C)=O"


@pytest.fixture(scope="module")
def client():
    # Context manager runs the lifespan handler (loads models + explainers once).
    with TestClient(app) as c:
        yield c


def test_predict_valid(client):
    r = client.post("/predict", json={"smiles": ASPIRIN})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"smiles", "structure", "solubility", "bbbp", "tox21"}
    assert len(body["tox21"]) == 12
    assert len(body["structure"]["atoms"]) == 13
    assert 0.0 <= body["bbbp"]["prediction"] <= 1.0


def test_predict_invalid_smiles_is_422(client):
    r = client.post("/predict", json={"smiles": "not_a_real_smiles!!!"})
    assert r.status_code == 422
    assert "not a valid SMILES" in r.json()["detail"]


def test_missing_field_is_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_oversized_smiles_is_rejected_before_featurization(client):
    response = client.post("/predict", json={"smiles": "C" * 4097})
    assert response.status_code == 422


def test_echoes_input_smiles_unchanged(client):
    # Atom indices are keyed to the submitted string's parse order, so it must not be canonicalized.
    r = client.post("/predict", json={"smiles": NON_CANONICAL_ASPIRIN})
    assert r.json()["smiles"] == NON_CANONICAL_ASPIRIN


def test_contribution_indices_refer_to_real_atoms(client):
    body = client.post("/predict", json={"smiles": ASPIRIN}).json()
    n_atoms = len(body["structure"]["atoms"])
    for result in [body["solubility"], body["bbbp"], *body["tox21"].values()]:
        assert all(0 <= int(i) < n_atoms for i in result["atom_contributions"])


def test_stateless_same_input_same_output(client):
    a = client.post("/predict", json={"smiles": ASPIRIN}).json()
    client.post("/predict", json={"smiles": "CCO"})
    b = client.post("/predict", json={"smiles": ASPIRIN}).json()
    assert a == b
