import numpy as np

from ml.featurize import FINGERPRINT_N_BITS, featurize_batch, featurize_smiles

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_fingerprint_shape_and_dtype():
    result = featurize_smiles(ASPIRIN)
    assert result is not None
    assert result.fingerprint.shape == (FINGERPRINT_N_BITS,)
    assert result.fingerprint.dtype == np.uint8
    assert set(np.unique(result.fingerprint)) <= {0, 1}


def test_invalid_smiles_returns_none():
    assert featurize_smiles("not_a_real_smiles!!!") is None


def test_deterministic():
    a = featurize_smiles(ASPIRIN)
    b = featurize_smiles(ASPIRIN)
    assert np.array_equal(a.fingerprint, b.fingerprint)


def test_bit_info_keys_are_exactly_the_on_bits():
    result = featurize_smiles(ASPIRIN)
    on_bits = set(np.flatnonzero(result.fingerprint).tolist())
    assert set(result.bit_info.keys()) == on_bits


def test_batch_matches_single_and_reports_invalid():
    smiles = [ASPIRIN, "garbage!!", "CCO"]
    X, valid, invalid = featurize_batch(smiles)
    assert valid == [0, 2]
    assert invalid == [1]
    assert np.array_equal(X[0], featurize_smiles(ASPIRIN).fingerprint)
    assert np.array_equal(X[1], featurize_smiles("CCO").fingerprint)
