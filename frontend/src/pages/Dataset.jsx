import { useEffect, useState } from "react";
import { API_URL, api } from "../api";

const EMPTY_OPTIONS = { remove_duplicates: true, fill_missing: true, remove_outliers: true, outlier_columns: [] };

function previewTable(columns, rows) {
  if (!rows?.length) return <div className="empty">No rows to preview.</div>;
  return <div className="table-wrap"><table className="data-table">
    <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
    <tbody>{rows.map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{row[column] == null ? "—" : String(row[column])}</td>)}</tr>)}</tbody>
  </table></div>;
}

export default function Dataset() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [dataset, setDataset] = useState(null);
  const [quality, setQuality] = useState(null);
  const [options, setOptions] = useState(EMPTY_OPTIONS);
  const [cleanPreview, setCleanPreview] = useState(null);

  async function loadDatasetAndQuality() {
    const current = await api("/dataset");
    const report = await api("/dataset/quality");
    setDataset(current);
    setQuality(report);
    setOptions({
      remove_duplicates: report.duplicate_rows > 0,
      fill_missing: report.missing_total > 0,
      remove_outliers: report.outlier_rows > 0,
      outlier_columns: Object.entries(report.outliers_by_column || {}).filter(([, count]) => count > 0).map(([column]) => column),
    });
    return current;
  }

  useEffect(() => {
    let alive = true;
    loadDatasetAndQuality().catch(() => {}).then(() => { if (!alive) return; });
    return () => { alive = false; };
  }, []);

  function handleChoose(event) {
    const chosen = event.target.files?.[0] || null;
    setFile(chosen);
    setError("");
    setMessage("");
    setCleanPreview(null);
  }

  async function handleUpload() {
    if (!file) { setError("Please choose a CSV file first."); return; }
    if (!file.name.toLowerCase().endsWith(".csv")) { setError("Only .csv files are supported."); return; }
    setUploading(true); setError(""); setMessage(""); setCleanPreview(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const uploaded = await api("/upload-dataset", { method: "POST", form, timeout: 60000 });
      localStorage.removeItem(`automl-pinned-charts:${localStorage.getItem("automl-chat-session") || "default"}`);
      window.dispatchEvent(new CustomEvent("analysis:clear-charts"));
      window.dispatchEvent(new CustomEvent("dataset:changed"));
      await loadDatasetAndQuality();
      setMessage(`${uploaded.filename} uploaded: ${uploaded.rows} rows, ${uploaded.columns} columns. Review the cleaning suggestions below.`);
    } catch (e) { setError(e.message); }
    finally { setUploading(false); }
  }

  async function handleCleanPreview() {
    setCleaning(true); setError(""); setCleanPreview(null);
    try { setCleanPreview(await api("/dataset/clean-preview", { method: "POST", body: options })); }
    catch (e) { setError(e.message); }
    finally { setCleaning(false); }
  }

  async function applyCleaning() {
    setCleaning(true); setError("");
    try {
      const result = await api("/dataset/clean", { method: "POST", body: options });
      localStorage.removeItem(`automl-pinned-charts:${localStorage.getItem("automl-chat-session") || "default"}`);
      window.dispatchEvent(new CustomEvent("analysis:clear-charts"));
      window.dispatchEvent(new CustomEvent("dataset:changed"));
      await loadDatasetAndQuality();
      setCleanPreview(null);
      setMessage(`Cleaned dataset applied: ${result.summary.rows_after} rows remain. The original upload is preserved. Analysis now uses the cleaned data; train a model again before making predictions.`);
    } catch (e) { setError(e.message); }
    finally { setCleaning(false); }
  }

  async function downloadDataset() {
    setDownloading(true); setError("");
    try {
      const sessionId = localStorage.getItem("automl-chat-session");
      const response = await fetch(`${API_URL}/dataset/download`, { headers: sessionId ? { "X-Chat-Session": sessionId } : {} });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Download failed (${response.status})`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `cleaned_${dataset?.filename || "dataset.csv"}`;
      document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
    } catch (e) { setError(e.message); }
    finally { setDownloading(false); }
  }

  const columns = dataset?.column_names || [];
  const hasCleaningWork = Boolean(quality && (quality.duplicate_rows || quality.missing_total || quality.outlier_rows));

  return <div className="dataset-manager">
    <div className="upload-card">
      <div className="upload-icon">↑</div>
      <h3>Upload Dataset</h3>
      <p>Select a CSV file to inspect, clean, preview, and use in this chat's AutoML workspace.</p>
      <input type="file" accept=".csv,text/csv" onChange={handleChoose} disabled={uploading || cleaning} />
      {file && <div className="selected-file">Selected: {file.name}</div>}
      <button className="primary-button" onClick={handleUpload} disabled={uploading || cleaning || !file}>{uploading ? "Uploading…" : "Upload Dataset"}</button>
      {uploading && <div className="alert info">Uploading dataset…</div>}
      {message && <div className="alert success">{message}</div>}
      {error && <div className="alert error">{error}</div>}
    </div>

    {dataset && <>
      <div className="stats-grid dataset-stats">
        <div className="stat-card"><span>DATASET</span><strong className="small-text">{dataset.filename}</strong><small>Current in this chat</small></div>
        <div className="stat-card"><span>ROWS</span><strong>{dataset.rows}</strong><small>Total records</small></div>
        <div className="stat-card"><span>COLUMNS</span><strong>{dataset.columns}</strong><small>Features + target</small></div>
      </div>

      {quality && <section className="panel cleaner-panel">
        <div className="cleaner-heading"><div><h3>Inspect and clean data</h3><p className="sub">Review the detected issues, preview the changes, then apply them. Outliers use the 1.5× IQR rule on selected numeric columns.</p></div>
          <div className="quality-pills"><span>{quality.missing_total} missing</span><span>{quality.duplicate_rows} duplicates</span><span>{quality.outlier_rows} outlier rows</span></div>
        </div>
        {hasCleaningWork ? <>
          <div className="clean-options">
            <label><input type="checkbox" checked={options.remove_duplicates} onChange={(e) => { setCleanPreview(null); setOptions((old) => ({ ...old, remove_duplicates: e.target.checked })); }} /> Remove duplicate rows <small>({quality.duplicate_rows})</small></label>
            <label><input type="checkbox" checked={options.fill_missing} onChange={(e) => { setCleanPreview(null); setOptions((old) => ({ ...old, fill_missing: e.target.checked })); }} /> Fill missing values <small>numeric: median · text: most common value</small></label>
            <label><input type="checkbox" checked={options.remove_outliers} onChange={(e) => { setCleanPreview(null); setOptions((old) => ({ ...old, remove_outliers: e.target.checked })); }} /> Remove rows with outliers</label>
          </div>
          {quality.numeric_columns.length > 0 && <div className="outlier-columns"><strong>Check numeric columns for outliers</strong><div>{quality.numeric_columns.map((column) => <label key={column}><input type="checkbox" disabled={!options.remove_outliers} checked={options.outlier_columns.includes(column)} onChange={(e) => { setCleanPreview(null); setOptions((old) => ({ ...old, outlier_columns: e.target.checked ? [...old.outlier_columns, column] : old.outlier_columns.filter((name) => name !== column) })); }} /> {column} <small>({quality.outliers_by_column[column] || 0})</small></label>)}</div></div>}
          <div className="form-actions"><button className="primary-button" onClick={handleCleanPreview} disabled={cleaning}>{cleaning && !cleanPreview ? "Preparing preview…" : "Preview cleaning"}</button><button className="secondary-button" onClick={downloadDataset} disabled={downloading}>{downloading ? "Preparing CSV…" : "Download current CSV"}</button></div>
          {cleanPreview && <div className="clean-preview">
            <div className="clean-preview-heading"><div><h4>Cleaned data preview</h4><p className="sub">No changes are applied until you confirm.</p></div><div className="preview-counts"><span>{cleanPreview.summary.rows_before} → {cleanPreview.summary.rows_after} rows</span><span>{cleanPreview.summary.duplicates_removed} duplicates removed</span><span>{cleanPreview.summary.outlier_rows_removed} outlier rows removed</span><span>{cleanPreview.summary.missing_values_filled} missing cells filled</span></div></div>
            {previewTable(cleanPreview.column_names, cleanPreview.preview)}
            <div className="form-actions"><button className="primary-button" onClick={applyCleaning} disabled={cleaning}>{cleaning ? "Applying…" : "Apply cleaning"}</button><button className="secondary-button" onClick={() => setCleanPreview(null)} disabled={cleaning}>Cancel</button></div>
          </div>}
        </> : <div className="clean-all-clear"><span>✓</span><div><strong>No common data quality issues detected</strong><small>You can still download the current CSV.</small></div><button className="secondary-button" onClick={downloadDataset} disabled={downloading}>{downloading ? "Preparing CSV…" : "Download current CSV"}</button></div>}
      </section>}

      <section className="panel dataset-preview-panel"><h3>Current dataset</h3><p className="sub">First 10 rows</p>{previewTable(columns, dataset.preview)}</section>
    </>}
  </div>;
}
