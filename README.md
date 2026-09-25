# ADMET Prediction

Paste a molecule's SMILES string and get predictions for **solubility**,
**blood-brain barrier (BBB) permeability**, and **toxicity** (12 Tox21 assays),
plus a SHAP-based explanation showing which atoms pushed each prediction up or down.

The app is fully stateless: no database, no accounts, no stored history. Every
request is computed from scratch from the SMILES in the request body.

## Quick start (Docker)

The trained models and datasets are not stored in git (too large), so generate them first:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv/Scripts/python scripts/download_data.py
.venv/Scripts/python -m scripts.train
```

Then:

```bash
docker compose up --build
```

Open http://localhost:8080. The backend takes ~20 seconds to start (it builds 14
SHAP explainers); the frontend container waits for its health check.

## Local development

```bash
# backend (from the project root)
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m uvicorn api.main:app --reload --port 8000

# frontend (second terminal)
cd frontend && npm install && npm run dev      # http://localhost:5173

# tests
.venv/Scripts/python -m pytest tests
```

The models must exist before the backend or tests will run; see the training
commands under "Quick start" above.

## How it works

```
SMILES -> ml/featurize.py -> 2048-bit Morgan fingerprint (+ bit -> atom map)
       -> Random Forest models (ESOL, BBBP, 12x Tox21)
       -> SHAP TreeExplainer -> per-bit scores -> per-atom scores via the bit -> atom map
       -> JSON (predictions, atom scores, 2D layout) -> React viewer
```

- **One featurization function** (`ml/featurize.py`) is used by both training and
  inference, so the model can never be scored on features computed differently
  from how it was trained.
- **SHAP explanations** credit each fingerprint bit's contribution to the single
  atom at the center of the substructure that set it (a deliberate simplification).
- **Chemistry stays in the backend**, including the 2D layout, so the frontend
  needs no chemistry library.

## API

`POST /predict` with `{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}` returns:

| field | contents |
|---|---|
| `smiles` | the input, echoed unchanged (atom indices refer to this exact string's parse order) |
| `structure` | `atoms` (index, element, x, y) and `bonds` (begin_atom, end_atom, order) |
| `solubility` | `prediction` (log mol/L), `base_value`, `atom_contributions` |
| `bbbp` | `prediction` (probability of permeable), `base_value`, `atom_contributions` |
| `tox21` | one entry per assay (e.g. `NR-AR`), each with `prediction` (probability of a positive result), `base_value`, `atom_contributions` |

`atom_contributions` maps atom index to a signed score: positive pushed the
prediction up, negative pushed it down. An invalid SMILES returns `422`.

## Model performance

Random Forest baselines (100 trees, ECFP4 fingerprints), evaluated with the same
split type the original MoleculeNet paper used for each dataset.

| Dataset | Metric | Ours | Closest published reference |
|---|---|---|---|
| ESOL (solubility) | RMSE, lower is better | **1.135** | 0.99 (XGBoost on ECFP); 0.58 (MPNN, deep) |
| BBBP (BBB permeability) | AUROC, scaffold split | **0.872** | 0.729 (Kernel SVM on ECFP) |
| Tox21 (toxicity) | mean AUROC over 12 tasks | **0.805** | 0.822 (Kernel SVM on ECFP); 0.829 (Graph Conv) |

Read these honestly:
- The reference numbers come from Wu et al. 2018 (MoleculeNet). That paper does not
  report a Random Forest score for these datasets, so each reference is the best
  *shallow* method it lists, not an RF-vs-RF comparison.
- **ESOL is weaker than the reference** (a plain RF on 1,128 molecules); deep models
  are far ahead.
- **Tox21 is slightly below** the reference, and **BBBP is above** it.
- Predictions are a research baseline, not a substitute for lab measurement.

## Project layout

```
api/        FastAPI app (single POST /predict endpoint)
ml/         featurize.py, explain.py, depict.py, splitting.py, models/*.pkl
scripts/    dataset download/exploration and model training
tests/      pytest suite (featurization, SHAP additivity, API)
frontend/   React (Vite) app
```

See `architecture.md` for design decisions and the reasoning behind them.
