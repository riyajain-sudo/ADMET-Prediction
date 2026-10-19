import { useState } from "react";
import { predict } from "./api";
import MoleculeViewer from "./MoleculeViewer";
import PredictionCard from "./PredictionCard";
import "./App.css";

const EXAMPLE_SMILES = "CC(=O)Oc1ccccc1C(=O)O"; // aspirin

function formatSolubility(value) {
  return `${value.toFixed(2)} log(mol/L)`;
}

function formatProbability(value) {
  return `${(value * 100).toFixed(0)}%`;
}

// Sorted worst-first so the assay most likely to flag as toxic is the
// default one shown, rather than an arbitrary/alphabetical first task.
function sortedTox21Tasks(tox21) {
  return Object.entries(tox21).sort((a, b) => b[1].prediction - a[1].prediction);
}

export default function App() {
  const [smiles, setSmiles] = useState(EXAMPLE_SMILES);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeProperty, setActiveProperty] = useState("solubility");
  const [activeTox21Task, setActiveTox21Task] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await predict(smiles);
      setResult(data);
      setActiveProperty("solubility");
      setActiveTox21Task(sortedTox21Tasks(data.tox21)[0][0]);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const activeContributions = result
    ? activeProperty === "solubility"
      ? result.solubility.atom_contributions
      : activeProperty === "bbbp"
        ? result.bbbp.atom_contributions
        : result.tox21[activeTox21Task].atom_contributions
    : null;
  const activeResult = result
    ? activeProperty === "solubility"
      ? result.solubility
      : activeProperty === "bbbp"
        ? result.bbbp
        : result.tox21[activeTox21Task]
    : null;

  return (
    <div>
      <h1>ADMET Prediction</h1>
      <p className="subtitle">
        Paste a molecule's SMILES string to predict solubility, blood-brain
        barrier permeability, and toxicity — with approximate atom-level
        allocation of mapped fingerprint contributions.
      </p>

      <form onSubmit={handleSubmit} className="smiles-form">
        <input
          type="text"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          placeholder="Enter a SMILES string, e.g. CC(=O)Oc1ccccc1C(=O)O"
          spellCheck={false}
        />
        <button type="submit" disabled={loading || !smiles.trim()}>
          {loading ? "Predicting…" : "Predict"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <>
          <MoleculeViewer structure={result.structure} contributions={activeContributions}
            unmappedContribution={activeResult?.unmapped_contribution} />

          <div className="prediction-cards">
            <PredictionCard
              title="Solubility"
              valueText={formatSolubility(result.solubility.prediction)}
              description="Predicted aqueous solubility (log scale); point estimate, no interval provided."
              active={activeProperty === "solubility"}
              onClick={() => setActiveProperty("solubility")}
            />
            <PredictionCard
              title="BBB Permeability"
              valueText={formatProbability(result.bbbp.prediction)}
              description="Model score (not a calibrated probability) for BBB permeability."
              active={activeProperty === "bbbp"}
              onClick={() => setActiveProperty("bbbp")}
            />
            <PredictionCard
              title="Toxicity"
              valueText={formatProbability(result.tox21[activeTox21Task].prediction)}
              description="Model score (not a calibrated probability) for this assay."
              active={activeProperty === "tox21"}
              onClick={() => setActiveProperty("tox21")}
            >
              <select
                value={activeTox21Task}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => {
                  setActiveTox21Task(e.target.value);
                  setActiveProperty("tox21");
                }}
              >
                {sortedTox21Tasks(result.tox21).map(([task, r]) => (
                  <option key={task} value={task}>
                    {task} ({formatProbability(r.prediction)})
                  </option>
                ))}
              </select>
            </PredictionCard>
          </div>
        </>
      )}
    </div>
  );
}
