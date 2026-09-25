"""
Step 2: Explore the three raw datasets before we build anything on top of them.

For each dataset we check the things that would otherwise bite us later:
  - row count and column names
  - which column holds the SMILES string, and which hold the target(s)
  - target distribution (class balance for classification, spread for regression)
  - missing values
  - whether RDKit can actually parse every SMILES string (invalid ones must be
    dropped now, not discovered mid-training in Step 4)

Run:
    .venv/Scripts/python.exe scripts/explore_datasets.py
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# RDKit logs a warning to stderr for every unparseable SMILES by default;
# we count failures ourselves and print a clean summary instead.
RDLogger.DisableLog("rdApp.*")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def check_smiles_validity(smiles_series: pd.Series) -> tuple[int, list[str]]:
    """Return (num_valid, list_of_invalid_smiles)."""
    invalid = []
    valid_count = 0
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(smi)
        else:
            valid_count += 1
    return valid_count, invalid


def explore_esol() -> None:
    print("\n" + "=" * 70)
    print("ESOL - water solubility (regression)")
    print("=" * 70)
    df = pd.read_csv(RAW_DIR / "esol.csv")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")

    target_col = "measured log solubility in mols per litre"
    smiles_col = "smiles"

    print(f"\ntarget column: '{target_col}'")
    print(df[target_col].describe())

    missing = df[[smiles_col, target_col]].isna().sum()
    print(f"\nmissing values:\n{missing}")

    valid, invalid = check_smiles_validity(df[smiles_col])
    print(f"\nSMILES parse check: {valid}/{len(df)} valid")
    if invalid:
        print(f"invalid SMILES (first 5): {invalid[:5]}")


def explore_bbbp() -> None:
    print("\n" + "=" * 70)
    print("BBBP - blood-brain barrier penetration (binary classification)")
    print("=" * 70)
    df = pd.read_csv(RAW_DIR / "bbbp.csv")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")

    target_col = "p_np"
    smiles_col = "smiles"

    print(f"\ntarget column: '{target_col}' (1 = permeates BBB, 0 = does not)")
    print(df[target_col].value_counts())
    print(f"class balance: {df[target_col].mean():.1%} positive")

    missing = df[[smiles_col, target_col]].isna().sum()
    print(f"\nmissing values:\n{missing}")

    valid, invalid = check_smiles_validity(df[smiles_col])
    print(f"\nSMILES parse check: {valid}/{len(df)} valid")
    if invalid:
        print(f"invalid SMILES (first 5): {invalid[:5]}")


def explore_tox21() -> None:
    print("\n" + "=" * 70)
    print("Tox21 - toxicity across 12 assays (multi-label classification)")
    print("=" * 70)
    df = pd.read_csv(RAW_DIR / "tox21.csv")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")

    smiles_col = "smiles"
    label_cols = [c for c in df.columns if c not in ("mol_id", "smiles")]
    print(f"\n{len(label_cols)} assay columns: {label_cols}")

    print("\nmissing values per assay (Tox21 is famously sparse - not every")
    print("molecule was tested in every assay):")
    print(df[label_cols].isna().sum())

    print("\npositive rate per assay (ignoring missing):")
    for col in label_cols:
        non_missing = df[col].dropna()
        print(f"  {col:20s} n={len(non_missing):5d}  positive={non_missing.mean():.1%}")

    valid, invalid = check_smiles_validity(df[smiles_col])
    print(f"\nSMILES parse check: {valid}/{len(df)} valid")
    if invalid:
        print(f"invalid SMILES (first 5): {invalid[:5]}")


def main() -> None:
    explore_esol()
    explore_bbbp()
    explore_tox21()


if __name__ == "__main__":
    main()
