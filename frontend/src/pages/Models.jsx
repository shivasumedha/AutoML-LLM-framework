import { useEffect, useState } from "react";
import { api } from "../api";

const MODEL_OPTIONS = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"];

function TrainingResults({ training }) {
  if (!training) return null;
  const results = training.results || [];
  return <div className="panel" style={{ marginTop: 16 }}>
    <div className="result-box" style={{ marginTop: 0, marginBottom: 18 }}>
      <span>ACTIVE MODEL · SAVED</span>
      <strong style={{ marginBottom: 4 }}>{training.selected_model || training.best_model}</strong>
      <div className="muted">Target: {training.target_column} · Best weighted F1: {training.best_f1_score ?? "—"} ({training.best_model || "metrics unavailable"})</div>
      {training.trained_dataset && <div className="muted">Trained on: {training.trained_dataset}</div>}
    </div>
    {results.length ? <>
      <h3>Model comparison</h3>
      <div className="table-wrap" style={{ marginTop: 12 }}><table className="data-table">
        <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Weighted F1</th><th>Active</th></tr></thead>
        <tbody>{results.map((r) => <tr key={r.model} className={r.model === training.selected_model ? "best" : ""}><td>{r.model}</td><td>{r.accuracy}</td><td>{r.precision}</td><td>{r.recall}</td><td>{r.f1_score}</td><td>{r.model === training.selected_model ? "✓" : ""}</td></tr>)}</tbody>
      </table></div>
      <h3 style={{ margin: "22px 0 8px" }}>Weighted F1 comparison</h3>
      {results.map((r) => <div className="bar-row" key={r.model}><span className="label">{r.model}</span><div className="bar-track"><div className="bar-fill" style={{ width: `${r.f1_score * 100}%` }} /></div><span className="value">{r.f1_score}</span></div>)}
    </> : <p className="muted" style={{ marginTop: 12 }}>This saved model is available for prediction. Its earlier comparison metrics were not saved.</p>}
  </div>;
}

export default function Models() {
  const [columns, setColumns] = useState([]);
  const [target, setTarget] = useState("");
  const [modelName, setModelName] = useState("Auto-select best");
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(null);
  const [showTraining, setShowTraining] = useState(false);

  useEffect(() => {
    let alive = true;
    Promise.all([api("/target-columns"), api("/model-status")])
      .then(([d, status]) => {
        if (!alive) return;
        setColumns(d.columns || []);
        const guess = [...(d.columns || [])].reverse().find((c) => c.unique_values >= 2 && c.unique_values <= 20);
        setTarget(status.available ? status.target_column : (guess || d.columns?.[d.columns.length - 1])?.name || "");
        setSaved(status.available ? status.training : null);
      })
      .catch((e) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => { alive = false; };
  }, []);

  async function handleTrain() {
    setTraining(true); setError("");
    try {
      await api("/train", { method: "POST", body: { target_column: target, model_name: modelName === "Auto-select best" ? null : modelName }, timeout: 300000 });
      const status = await api("/model-status");
      setSaved(status.training);
      setShowTraining(false);
      window.dispatchEvent(new CustomEvent("model:changed"));
    } catch (e) { setError(e.message); }
    finally { setTraining(false); }
  }

  if (loading) return <div className="empty">Loading saved model and dataset columns…</div>;
  if (!columns.length) return <div className="alert error wide">{error || "Upload a dataset first."}</div>;
  const selected = columns.find((c) => c.name === target);

  return <div>
    {saved && <TrainingResults training={saved} />}
    {(!saved || showTraining) ? <div className="panel" style={{ marginTop: saved ? 16 : 0 }}>
      <h3>{saved ? "Train a new model" : "Train models once for this dataset"}</h3>
      <p className="sub">The saved model stays available when you visit other sections. Training compares four classifiers; choose one to activate, or let the app select the highest weighted F1.</p>
      <div className="form-grid" style={{ maxWidth: 760 }}>
        <div className="field"><label htmlFor="target">Target column</label><select id="target" value={target} onChange={(e) => setTarget(e.target.value)} disabled={training}>{columns.map((c) => <option key={c.name} value={c.name}>{c.name} ({c.unique_values} unique)</option>)}</select>{selected && <small>Type: {selected.data_type}</small>}</div>
        <div className="field"><label htmlFor="model-choice">Model to activate</label><select id="model-choice" value={modelName} onChange={(e) => setModelName(e.target.value)} disabled={training}><option>Auto-select best</option>{MODEL_OPTIONS.map((name) => <option key={name}>{name}</option>)}</select><small>Choose SVM or another classifier, or let weighted F1 pick.</small></div>
      </div>
      <div className="form-actions"><button className="primary-button" onClick={handleTrain} disabled={training || !target}>{training ? "Training…" : "Train and save model"}</button>{saved && <button className="secondary-button" onClick={() => setShowTraining(false)} disabled={training}>Cancel</button>}{training && <span className="muted">This can take a little while.</span>}</div>
      {error && <div className="alert error" style={{ margin: "16px 0 0", maxWidth: "none" }}>{error}</div>}
    </div> : <button className="secondary-button" style={{ marginTop: 16 }} onClick={() => setShowTraining(true)}>Train another model</button>}
  </div>;
}
