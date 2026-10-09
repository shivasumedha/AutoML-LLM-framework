import { useEffect, useState } from "react";
import { api } from "../api";

export default function Prediction({ onNavigate }) {
  const [meta, setMeta] = useState(null);
  const [values, setValues] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    let alive = true;
    api("/features")
      .then((d) => {
        if (!alive) return;
        setMeta(d);
        const initial = {};
        d.features.forEach((f) => {
          initial[f.name] = f.type === "number" ? (f.median ?? "") : (f.options[0] ?? "");
        });
        setValues(initial);
      })
      .catch((e) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  async function handlePredict() {
    setPredicting(true);
    setError("");
    setResult(null);
    try {
      const data = {};
      meta.features.forEach((f) => {
        const v = values[f.name];
        data[f.name] = f.type === "number" ? (v === "" ? null : Number(v)) : v;
      });
      setResult(await api("/predict", { method: "POST", body: { data } }));
    } catch (e) {
      setError(e.message);
    } finally {
      setPredicting(false);
    }
  }

  if (loading) return <div className="empty">Loading model...</div>;
  if (!meta) return <div className="panel"><h3>No saved model is available</h3><p className="sub">Train a model once for the current dataset. It will remain available when you move between sections.</p>{error && <div className="alert error" style={{ margin: "12px 0", maxWidth: "none" }}>{error}</div>}<button className="primary-button" onClick={() => onNavigate?.("models")}>Open Models</button></div>;

  return (
    <div className="panel">
      <h3>Predict {meta.target_column}</h3>
      <p className="sub">Using saved model: {meta.model}. This section loads the existing model; it does not train again. Defaults are the median / first value of each column.</p>

      <div className="form-grid">
        {meta.features.map((f) => (
          <div className="field" key={f.name}>
            <label htmlFor={f.name}>{f.name}</label>
            {f.type === "number" ? (
              <>
                <input
                  id={f.name}
                  type="number"
                  step="any"
                  value={values[f.name]}
                  onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
                />
                <small>Range: {f.min} – {f.max}</small>
              </>
            ) : (
              <select id={f.name} value={values[f.name]} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}>
                {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            )}
          </div>
        ))}
      </div>

      <div className="form-actions">
        <button className="primary-button" onClick={handlePredict} disabled={predicting}>
          {predicting ? "Predicting..." : "Predict"}
        </button>
      </div>

      {error && <div className="alert error" style={{ margin: "16px 0 0", maxWidth: "none" }}>{error}</div>}

      {result && (
        <div className="result-box">
          <span>PREDICTION</span>
          <strong>{String(result.prediction)}</strong>
          {result.probabilities?.map((p) => (
            <div className="bar-row" key={p.class}>
              <span className="label">{p.class}</span>
              <div className="bar-track"><div className="bar-fill" style={{ width: `${p.probability * 100}%` }} /></div>
              <span className="value">{(p.probability * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
