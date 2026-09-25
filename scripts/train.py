"""
Step 4: Train baseline Random Forest models for all three properties, and
report an honest comparison against the closest VERIFIED published numbers.

Model choice: Random Forest. It's the standard classical-ML baseline in
cheminformatics -- fast to train (no GPU), works well on the sparse binary
fingerprint features from Step 3, and pairs naturally with SHAP's
TreeExplainer in Step 5 (fast, exact SHAP values for tree ensembles, unlike
the slower approximate methods other model types require).

Split choice: matches what the original MoleculeNet paper (Wu et al. 2018,
Chemical Science) used per dataset -- scaffold split for BBBP, random split
for ESOL and Tox21. A different split would make the benchmark comparison
below meaningless (scaffold split is deliberately the harder test).

Benchmark honesty note: the original paper's own summary table does NOT
publish a Random-Forest-specific score for these three datasets -- it notes
RF results were only included for smaller datasets. The numbers below are
the closest VERIFIED reference points (fetched from the paper directly, not
from memory): the best *shallow* method the paper reports per dataset. They
are printed alongside our RF results as an honest ballpark check, not
claimed as an exact RF-vs-RF reproduction.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

from ml.featurize import featurize_batch
from ml.splitting import random_split, scaffold_split

DATA_DIR = Path("data/raw")
MODELS_DIR = Path("ml/models")
RANDOM_STATE = 42
# 500 trees was the initial default; a quick check on one Tox21 task showed
# 100 trees performs equally well (0.806 vs 0.794 AUROC) at ~1/5th the
# pickle size, so 100 is used everywhere for a smaller, equally-accurate
# model -- important since Tox21 trains 12 separate forests.
N_ESTIMATORS = 100


def train_esol() -> float:
    print("\n" + "=" * 70)
    print("ESOL - solubility (regression)")
    print("=" * 70)
    df = pd.read_csv(DATA_DIR / "esol.csv")
    target_col = "measured log solubility in mols per litre"

    X, valid_idx, invalid_idx = featurize_batch(df["smiles"].tolist())
    y = df[target_col].values[valid_idx]
    print(f"featurized {len(valid_idx)}/{len(df)} molecules ({len(invalid_idx)} dropped as invalid)")

    train_idx, test_idx = random_split(len(y), frac_train=0.8, seed=RANDOM_STATE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"random split: {len(train_idx)} train / {len(test_idx)} test")

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    print(f"\nour RF RMSE (lower is better): {rmse:.3f}")
    print("closest VERIFIED reference (MoleculeNet paper, best shallow method")
    print("  = XGBoost on ECFP): RMSE 0.99  (MPNN, a deep model, reaches 0.58)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "esol_rf.pkl")
    print(f"saved -> {MODELS_DIR / 'esol_rf.pkl'}")
    return rmse


def train_bbbp() -> float:
    print("\n" + "=" * 70)
    print("BBBP - blood-brain barrier penetration (binary classification)")
    print("=" * 70)
    df = pd.read_csv(DATA_DIR / "bbbp.csv")
    target_col = "p_np"

    X, valid_idx, invalid_idx = featurize_batch(df["smiles"].tolist())
    y = df[target_col].values[valid_idx]
    valid_smiles = df["smiles"].values[valid_idx].tolist()
    print(f"featurized {len(valid_idx)}/{len(df)} molecules ({len(invalid_idx)} dropped as invalid)")

    train_idx, test_idx = scaffold_split(valid_smiles, frac_train=0.8, seed=RANDOM_STATE)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"scaffold split: {len(train_idx)} train / {len(test_idx)} test")
    print(f"train positive rate: {y_train.mean():.1%}  |  test positive rate: {y_test.mean():.1%}")

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, proba))
    print(f"\nour RF AUROC (higher is better, 0.5=random, 1.0=perfect): {auroc:.3f}")
    print("closest VERIFIED reference (MoleculeNet paper, scaffold split,")
    print("  best shallow method = Kernel SVM on ECFP): AUROC 0.729")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "bbbp_rf.pkl")
    print(f"saved -> {MODELS_DIR / 'bbbp_rf.pkl'}")
    return auroc


def train_tox21() -> float:
    print("\n" + "=" * 70)
    print("Tox21 - toxicity across 12 assays (multi-label classification)")
    print("=" * 70)
    df = pd.read_csv(DATA_DIR / "tox21.csv")
    label_cols = [c for c in df.columns if c not in ("mol_id", "smiles")]

    X, valid_idx, invalid_idx = featurize_batch(df["smiles"].tolist())
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    print(f"featurized {len(valid_idx)}/{len(df)} molecules ({len(invalid_idx)} dropped as invalid)")

    train_idx, test_idx = random_split(len(df_valid), frac_train=0.8, seed=RANDOM_STATE)
    print(f"random split: {len(train_idx)} train / {len(test_idx)} test")
    print("(one split, reused across all 12 assay sub-models, for a consistent test set)")

    models: dict[str, RandomForestClassifier] = {}
    aurocs: dict[str, float] = {}

    for task in label_cols:
        labels = df_valid[task].values  # float, NaN where this assay wasn't tested

        # Per-task masking: only rows where THIS assay has a non-missing
        # label. Dropping any row missing ANY assay would throw away most
        # of the dataset (Step 2 found 7-26% missing per assay).
        task_train_idx = [i for i in train_idx if not np.isnan(labels[i])]
        task_test_idx = [i for i in test_idx if not np.isnan(labels[i])]

        y_train_task = labels[task_train_idx].astype(int)
        y_test_task = labels[task_test_idx].astype(int)

        if len(set(y_train_task)) < 2 or len(set(y_test_task)) < 2:
            print(f"  {task:20s} skipped (train or test has only one class present)")
            continue

        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X[task_train_idx], y_train_task)

        proba = model.predict_proba(X[task_test_idx])[:, 1]
        auroc = float(roc_auc_score(y_test_task, proba))
        aurocs[task] = auroc
        models[task] = model
        print(f"  {task:20s} AUROC={auroc:.3f}  (train n={len(task_train_idx)}, test n={len(task_test_idx)})")

    mean_auroc = float(np.mean(list(aurocs.values())))
    print(f"\nour RF mean AUROC across {len(aurocs)}/{len(label_cols)} tasks: {mean_auroc:.3f}")
    print("closest VERIFIED reference (MoleculeNet paper, random split,")
    print("  best shallow method = Kernel SVM on ECFP): AUROC 0.822")
    print("  (Graph Convolution, a deep model, reaches 0.829)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODELS_DIR / "tox21_rf.pkl")
    print(f"saved -> {MODELS_DIR / 'tox21_rf.pkl'} ({len(models)} per-task models in one dict)")
    return mean_auroc


def main() -> None:
    esol_rmse = train_esol()
    bbbp_auroc = train_bbbp()
    tox21_mean_auroc = train_tox21()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'dataset':<8} {'our RF':>10}   closest verified reference (NOT an RF-specific number)")
    print(f"{'ESOL':<8} {esol_rmse:>10.3f}   RMSE  | XGBoost 0.99, MPNN(deep) 0.58")
    print(f"{'BBBP':<8} {bbbp_auroc:>10.3f}   AUROC | Kernel SVM 0.729 (scaffold split)")
    print(f"{'Tox21':<8} {tox21_mean_auroc:>10.3f}   AUROC | Kernel SVM 0.822, GC(deep) 0.829 (random split)")
    print("\nReference numbers were fetched from the original MoleculeNet paper (Wu et al.")
    print("2018) just now, not recalled from memory. The paper's own summary table does")
    print("not include an RF-specific score for these three datasets, so treat the above")
    print("as an honest ballpark sanity check, not an exact apples-to-apples reproduction.")


if __name__ == "__main__":
    main()
