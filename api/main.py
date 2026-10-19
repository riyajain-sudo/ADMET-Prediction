"""
Step 6: the whole backend -- one stateless POST /predict endpoint.

A request carries one SMILES string; the response carries a solubility
prediction, a BBB-permeability prediction, all 12 Tox21 toxicity
predictions, and a SHAP explanation for each. Nothing is stored between
requests -- no database, no session, no per-user state. Every prediction is
computed fresh from the request body alone.

Models and SHAP explainers are built ONCE at startup (see `lifespan`
below), not per request. Loading a .pkl file and building a SHAP background
explainer has a fixed cost unrelated to which molecule a user submits --
paying that cost once at boot keeps each request cheap (just featurize one
molecule, call .predict()/.predict_proba(), and run SHAP against a
fingerprint that's already computed). This doesn't reintroduce state: no
request's outcome depends on any other request, only on these fixed,
read-only, boot-time-loaded models.
"""

from contextlib import asynccontextmanager
from typing import TypedDict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.depict import MoleculeLayout, compute_layout
from ml.explain import Explanation, explain_fingerprint, make_explainer, sample_background
from ml.featurize import featurize_smiles

MODELS_DIR = "ml/models"
DATA_DIR = "data/raw"


class Resources(TypedDict):
    esol_model: object
    esol_explainer: object
    bbbp_model: object
    bbbp_explainer: object
    tox21_models: dict
    tox21_explainers: dict


resources: Resources = {}  # type: ignore[typeddict-item]


@asynccontextmanager
async def lifespan(app: FastAPI):
    esol_model = joblib.load(f"{MODELS_DIR}/esol_rf.pkl")
    esol_background = sample_background(pd.read_csv(f"{DATA_DIR}/esol.csv")["smiles"].tolist())
    resources["esol_model"] = esol_model
    resources["esol_explainer"] = make_explainer(esol_model, esol_background)

    bbbp_model = joblib.load(f"{MODELS_DIR}/bbbp_rf.pkl")
    bbbp_background = sample_background(pd.read_csv(f"{DATA_DIR}/bbbp.csv")["smiles"].tolist())
    resources["bbbp_model"] = bbbp_model
    resources["bbbp_explainer"] = make_explainer(bbbp_model, bbbp_background)

    tox21_models = joblib.load(f"{MODELS_DIR}/tox21_rf.pkl")
    # One shared background sample for all 12 tasks -- they're all trained
    # on the same tox21.csv molecules, just with different per-task labels.
    tox21_background = sample_background(pd.read_csv(f"{DATA_DIR}/tox21.csv")["smiles"].tolist())
    resources["tox21_models"] = tox21_models
    resources["tox21_explainers"] = {
        task: make_explainer(model, tox21_background) for task, model in tox21_models.items()
    }

    yield
    resources.clear()


app = FastAPI(title="ADMET Prediction", lifespan=lifespan)

# No cookies/auth/session exist in this app (stateless by design), so a
# permissive origin policy carries no meaningful risk -- it's what lets any
# browser-based frontend call this API at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    # Bound parsing/featurization work for untrusted requests while allowing
    # ordinary drug-like molecules with substantial stereochemical notation.
    smiles: str = Field(..., max_length=4096)


class PropertyResult(BaseModel):
    prediction: float
    base_value: float
    atom_contributions: dict[int, float]
    unmapped_contribution: float

    @staticmethod
    def from_explanation(explanation: Explanation) -> "PropertyResult":
        return PropertyResult(
            prediction=explanation.prediction,
            base_value=explanation.base_value,
            atom_contributions=explanation.atom_contributions,
            unmapped_contribution=explanation.unmapped_contribution,
        )


class AtomOut(BaseModel):
    index: int
    element: str
    x: float
    y: float


class BondOut(BaseModel):
    begin_atom: int
    end_atom: int
    order: float


class StructureOut(BaseModel):
    atoms: list[AtomOut]
    bonds: list[BondOut]

    @staticmethod
    def from_layout(layout: MoleculeLayout) -> "StructureOut":
        return StructureOut(
            atoms=[AtomOut(index=a.index, element=a.element, x=a.x, y=a.y) for a in layout.atoms],
            bonds=[BondOut(begin_atom=b.begin_atom, end_atom=b.end_atom, order=b.order) for b in layout.bonds],
        )


class PredictResponse(BaseModel):
    smiles: str
    structure: StructureOut
    solubility: PropertyResult
    bbbp: PropertyResult
    tox21: dict[str, PropertyResult]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    featurized = featurize_smiles(request.smiles)
    if featurized is None:
        raise HTTPException(status_code=422, detail=f"'{request.smiles}' is not a valid SMILES string")

    solubility = explain_fingerprint(resources["esol_explainer"], featurized, is_classifier=False)
    bbbp = explain_fingerprint(resources["bbbp_explainer"], featurized, is_classifier=True)
    tox21 = {
        task: explain_fingerprint(explainer, featurized, is_classifier=True)
        for task, explainer in resources["tox21_explainers"].items()
    }

    return PredictResponse(
        # Echo the exact input, NOT a canonicalized form: atom_contributions'
        # indices are keyed to THIS string's parse order, and RDKit's
        # canonical SMILES can reorder atoms relative to the input. If the
        # frontend re-parses whatever SMILES comes back here, it must be
        # this exact string for the atom indices to still line up.
        smiles=request.smiles,
        structure=StructureOut.from_layout(compute_layout(featurized.mol)),
        solubility=PropertyResult.from_explanation(solubility),
        bbbp=PropertyResult.from_explanation(bbbp),
        tox21={task: PropertyResult.from_explanation(explanation) for task, explanation in tox21.items()},
    )
