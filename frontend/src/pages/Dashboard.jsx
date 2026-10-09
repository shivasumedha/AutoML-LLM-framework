import { useEffect, useState } from "react";
import { api } from "../api";

export default function Dashboard({ onNavigate }) {
  const [dataset, setDataset] = useState(null);
  const [model, setModel] = useState(null);

  useEffect(() => {
    let alive = true;
    api("/dataset")
      .then((d) => alive && setDataset(d))
      .catch(() => {});
    api("/model-status")
      .then((status) => alive && setModel(status.available ? status : null))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const cards = [
    { id: "dataset", icon: "↑", title: "Upload Dataset", text: "Upload a CSV file and preview your data.", cta: "Open Dataset →" },
    { id: "analysis", icon: "▤", title: "Analyze Data", text: "Statistics, missing values, distributions and correlations.", cta: "Open Analysis →" },
    { id: "models", icon: "◆", title: model ? "Saved Model" : "Train First Model", text: model ? "Review saved model results. Training again is optional." : "Choose a target and train once to enable predictions.", cta: model ? "Review results →" : "Train a model →" },
    { id: "prediction", icon: "✦", title: "Predict", text: "Use the saved model to make predictions. Section changes do not retrain it.", cta: "Open Prediction →" },
  ];

  return (
    <div>
      <div className="hero-card">
        <div>
          <span className="badge">AUTOML + LLM</span>
          <h2>Build models<br />from your data.</h2>
          <p>
            Upload a CSV, explore it, train several models automatically, and ask the assistant
            questions about your dataset.
          </p>
          <button className="primary-button" onClick={() => onNavigate("dataset")}>
            {dataset ? "Change Dataset" : "Upload Dataset"}
          </button>
        </div>
        <div className="hero-visual">
          <div className="orbit" />
          <div className="orbit orbit-two" />
          <div className="orb">AI</div>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span>DATASET</span>
          <strong className="small-text">{dataset ? dataset.filename : "None"}</strong>
          <small>{dataset ? "Loaded" : "Upload a CSV to begin"}</small>
        </div>
        <div className="stat-card">
          <span>ROWS</span>
          <strong>{dataset ? dataset.rows : "—"}</strong>
          <small>Total records</small>
        </div>
        <div className="stat-card">
          <span>COLUMNS</span>
          <strong>{dataset ? dataset.columns : "—"}</strong>
          <small>Features + target</small>
        </div>
        <div className="stat-card">
          <span>MODEL</span>
          <strong className="small-text">{model ? model.model : "Not trained"}</strong>
          <small>{model ? `Target: ${model.target_column}` : "No saved model yet"}</small>
        </div>
      </div>

      <div className="quick-grid">
        {cards.map((c) => (
          <button key={c.id} className="feature-card" onClick={() => onNavigate(c.id)}>
            <div className="feature-icon">{c.icon}</div>
            <h3>{c.title}</h3>
            <p>{c.text}</p>
            <span>{c.cta}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
