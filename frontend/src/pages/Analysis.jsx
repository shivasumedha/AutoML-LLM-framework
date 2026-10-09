import { useEffect, useState } from "react";
import { api } from "../api";

function Histogram({ name, bins }) {
  const max = Math.max(...bins.map((b) => b.count), 1);
  return (
    <div className="panel">
      <h3>{name}</h3>
      <p className="sub">Distribution</p>
      <div className="hist">
        {bins.map((b) => (
          <div key={b.range} title={`${b.range}: ${b.count}`} style={{ height: `${(b.count / max) * 100}%` }} />
        ))}
      </div>
      <div className="hist-axis">
        <span>{bins[0].range.split(" - ")[0]}</span>
        <span>{bins[bins.length - 1].range.split(" - ")[1]}</span>
      </div>
    </div>
  );
}

function Bars({ name, items }) {
  const max = Math.max(...items.map((i) => i.count), 1);
  return (
    <div className="panel">
      <h3>{name}</h3>
      <p className="sub">Top values</p>
      {items.map((i) => (
        <div className="bar-row" key={i.name}>
          <span className="label" title={i.name}>{i.name}</span>
          <div className="bar-track"><div className="bar-fill" style={{ width: `${(i.count / max) * 100}%` }} /></div>
          <span className="value">{i.count}</span>
        </div>
      ))}
    </div>
  );
}

function PinnedChart({ chart }) {
  if (chart.type === "scatter") {
    const points = chart.points || [];
    const xs = points.map((p) => Number(p.x)); const ys = points.map((p) => Number(p.y));
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    return <div className="panel"><h3>{chart.column} by {chart.group_by}</h3><svg viewBox="0 0 600 220" role="img" aria-label="Scatter plot">{points.map((p, i) => <circle key={i} cx={30 + ((p.x - minX) / (maxX - minX || 1)) * 540} cy={200 - ((p.y - minY) / (maxY - minY || 1)) * 180} r="3" fill="var(--green-light)" />)}</svg></div>;
  }
  const points = chart.points || [];
  const max = Math.max(...points.map((p) => p.value), 1);
  return <div className="panel"><h3>{chart.type === "histogram" ? `${chart.column} distribution` : `${chart.column} counts`}</h3><div className="chat-chart-bars">{points.map((p, i) => <div className="chat-chart-column" key={`${p.label}-${p.series || i}`} title={`${p.label}${p.series ? ` · ${p.series}` : ""}: ${p.value}`}><span>{p.value}</span><i style={{ height: `${Math.max(3, (p.value / max) * 100)}%` }} /><small>{p.series ? `${p.label} · ${p.series}` : p.label}</small></div>)}</div></div>;
}

export default function Analysis({ sessionId, onNavigate }) {
  const [info, setInfo] = useState(null);
  const [charts, setCharts] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [versions, setVersions] = useState([]);
  const [pinned, setPinned] = useState(() => {
    try { return JSON.parse(localStorage.getItem(`automl-pinned-charts:${sessionId || "default"}`) || "[]"); } catch { return []; }
  });

  useEffect(() => {
    let alive = true;
    const load = () => {
      setError("");
      setLoading(true);
      return Promise.all([api("/analyze"), api("/charts"), api("/dataset-history")])
      .then(([a, c, h]) => {
        if (!alive) return;
        setInfo(a);
        setCharts(c);
        setVersions(h.versions || []);
      })
      .catch((e) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    };
    const pinChart = (event) => setPinned((old) => {
      const next = [event.detail, ...old.filter((c) => !(c.type === event.detail.type && c.column === event.detail.column))].slice(0, 8);
      localStorage.setItem(`automl-pinned-charts:${sessionId || "default"}`, JSON.stringify(next));
      return next;
    });
    const clearPinnedCharts = () => {
      setPinned([]);
      localStorage.removeItem(`automl-pinned-charts:${sessionId || "default"}`);
    };
    load();
    window.addEventListener("dataset:changed", load);
    window.addEventListener("analysis:pin-chart", pinChart);
    window.addEventListener("analysis:clear-charts", clearPinnedCharts);
    return () => {
      alive = false;
      window.removeEventListener("dataset:changed", load);
      window.removeEventListener("analysis:pin-chart", pinChart);
      window.removeEventListener("analysis:clear-charts", clearPinnedCharts);
    };
  }, [sessionId]);

  if (loading) return <div className="empty">Analyzing dataset...</div>;
  if (error) return error.toLowerCase().includes("upload a dataset") || error.toLowerCase().includes("dataset first")
    ? <div className="panel empty-workspace"><h3>No analysis in this chat yet</h3><p className="sub">Upload a dataset in this chat to create fresh analysis.</p><button className="primary-button" onClick={() => onNavigate?.("dataset")}>Upload dataset</button></div>
    : <div className="alert error wide">{error}</div>;

  const totalMissing = Object.values(info.missing_values).reduce((a, b) => a + b, 0);
  const hist = Object.entries(charts.histograms).slice(0, 8);
  const cats = Object.entries(charts.categories).slice(0, 6);
  const corrCols = [...new Set(charts.correlation.map((c) => c.x))];
  const corrMap = Object.fromEntries(charts.correlation.map((c) => [`${c.x}|${c.y}`, c.value]));
  const statNames = ["count", "mean", "std", "min", "50%", "max"];
  const numCols = Object.keys(info.numeric_summary);
  const missingMax = Math.max(...Object.values(info.missing_values), 1);

  return (
    <div>
      {versions.length > 0 && <div className="panel version-panel"><h3>Dataset versions</h3><p className="sub">Chat edits save new versions. Say “undo that” in chat to restore the previous one.</p><div className="version-list">{versions.slice(0, 6).map((v) => <span key={v.filename} className={v.current ? "version-current" : ""}>{v.current ? "Current · " : ""}{v.filename}</span>)}</div></div>}
      <div className="stats-grid">
        <div className="stat-card"><span>ROWS</span><strong>{info.rows}</strong><small>{info.filename}</small></div>
        <div className="stat-card"><span>COLUMNS</span><strong>{info.columns}</strong><small>{info.numeric_columns.length} numeric · {info.categorical_columns.length} categorical</small></div>
        <div className="stat-card"><span>MISSING VALUES</span><strong>{totalMissing}</strong><small>Across all columns</small></div>
        <div className="stat-card"><span>DUPLICATES</span><strong>{info.duplicate_rows}</strong><small>Duplicate rows</small></div>
      </div>

      {pinned.length > 0 && <><div className="section-heading" style={{ marginTop: 28 }}><h2>Charts from AI Chat</h2></div><div className="grid-2">{pinned.map((c, i) => <PinnedChart key={`${c.type}-${c.column}-${i}`} chart={c} />)}</div></>}

      {hist.length > 0 && (
        <>
          <div className="section-heading" style={{ marginTop: 28 }}><h2>Distributions</h2></div>
          <div className="grid-2">{hist.map(([n, b]) => <Histogram key={n} name={n} bins={b} />)}</div>
        </>
      )}

      {cats.length > 0 && (
        <>
          <div className="section-heading" style={{ marginTop: 28 }}><h2>Categories</h2></div>
          <div className="grid-2">{cats.map(([n, items]) => <Bars key={n} name={n} items={items} />)}</div>
        </>
      )}

      <div className="section-heading" style={{ marginTop: 28 }}><h2>Missing values</h2></div>
      <div className="panel">
        {Object.entries(info.missing_values).slice(0, 12).map(([col, n]) => (
          <div className="bar-row" key={col}>
            <span className="label">{col}</span>
            <div className="bar-track"><div className="bar-fill" style={{ width: `${(n / missingMax) * 100}%` }} /></div>
            <span className="value">{n}</span>
          </div>
        ))}
      </div>

      {numCols.length > 0 && (
        <>
          <div className="section-heading" style={{ marginTop: 28 }}><h2>Numeric summary</h2></div>
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>Column</th>{statNames.map((s) => <th key={s}>{s}</th>)}</tr></thead>
              <tbody>
                {numCols.map((c) => (
                  <tr key={c}>
                    <td>{c}</td>
                    {statNames.map((s) => {
                      const v = info.numeric_summary[c][s];
                      return <td key={s}>{v === null || v === undefined ? "—" : Number(v).toFixed(2)}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {corrCols.length >= 2 && (
        <>
          <div className="section-heading" style={{ marginTop: 28 }}><h2>Correlation</h2></div>
          <div className="table-wrap" style={{ padding: 14 }}>
            <table className="heat">
              <thead><tr><th />{corrCols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {corrCols.map((r) => (
                  <tr key={r}>
                    <th>{r}</th>
                    {corrCols.map((c) => {
                      const v = corrMap[`${r}|${c}`];
                      const bg = v === null || v === undefined ? "transparent" : v >= 0
                        ? `rgba(32,227,154,${Math.abs(v) * 0.85})`
                        : `rgba(255,107,107,${Math.abs(v) * 0.85})`;
                      return <td key={c} style={{ background: bg }}>{v === null || v === undefined ? "—" : v.toFixed(2)}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
