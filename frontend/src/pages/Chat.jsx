import { useEffect, useRef, useState } from "react";
import { api } from "../api";

const SUGGESTIONS = [
  "Summarize my dataset", "Which columns have missing values?", "Are there duplicates?",
  "Show the distribution of a numeric column", "Compare averages across groups",
  "Which columns are associated with my target?", "What should I fix before training?",
  "Fill missing values using the median",
];

function isTableSeparator(line) {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
}

function splitTableRow(line) {
  let value = line.trim();
  if (value.startsWith("|")) value = value.slice(1);
  if (value.endsWith("|") && !value.endsWith("\\|")) value = value.slice(0, -1);
  const cells = [];
  let cell = "";
  let escaped = false;
  for (const character of value) {
    if (character === "|" && !escaped) { cells.push(cell.trim().replace(/\\\|/g, "|")); cell = ""; }
    else cell += character;
    escaped = character === "\\" && !escaped;
    if (character !== "\\") escaped = false;
  }
  cells.push(cell.trim().replace(/\\\|/g, "|"));
  return cells;
}

function InlineText({ text }) {
  const pattern = /(\*\*[^*]+\*\*|~~[^~]+~~|`[^`]+`|\*[^*\n]+\*|<br\s*\/?\s*>)/gi;
  const pieces = String(text).split(pattern);
  return pieces.map((piece, index) => {
    if (!piece) return null;
    if (/^<br\s*\/?\s*>$/i.test(piece)) return <br key={index} />;
    if (piece.startsWith("**") && piece.endsWith("**")) return <strong key={index}><InlineText text={piece.slice(2, -2)} /></strong>;
    if (piece.startsWith("~~") && piece.endsWith("~~")) return <del key={index}>{piece.slice(2, -2)}</del>;
    if (piece.startsWith("`") && piece.endsWith("`")) return <code className="chat-inline-code" key={index}>{piece.slice(1, -1)}</code>;
    if (piece.startsWith("*") && piece.endsWith("*")) return <em key={index}>{piece.slice(1, -1)}</em>;
    return piece;
  });
}

function MessageContent({ text }) {
  const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const Tag = `h${heading[1].length}`;
      blocks.push(<Tag key={`h-${index}`}><InlineText text={heading[2]} /></Tag>);
      index += 1;
      continue;
    }

    if (/^\s*\|/.test(line) && index + 1 < lines.length && isTableSeparator(lines[index + 1])) {
      const headers = splitTableRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && /^\s*\|/.test(lines[index])) {
        const row = splitTableRow(lines[index]);
        if (!isTableSeparator(lines[index])) rows.push(row);
        index += 1;
      }
      blocks.push(<div className="chat-table-wrap" key={`table-${index}`}><table className="chat-table"><thead><tr>{headers.map((cell, cellIndex) => <th key={cellIndex}><InlineText text={cell} /></th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{headers.map((_, cellIndex) => <td key={cellIndex}><InlineText text={row[cellIndex] || ""} /></td>)}</tr>)}</tbody></table></div>);
      continue;
    }

    if (/^\s*```/.test(line)) {
      const code = [];
      index += 1;
      while (index < lines.length && !/^\s*```/.test(lines[index])) { code.push(lines[index]); index += 1; }
      index += 1;
      blocks.push(<pre className="chat-code-block" key={`code-${index}`}><code>{code.join("\n")}</code></pre>);
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quote = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) { quote.push(lines[index].replace(/^\s*>\s?/, "")); index += 1; }
      blocks.push(<blockquote key={`quote-${index}`}>{quote.map((part, quoteIndex) => <div key={quoteIndex}><InlineText text={part} /></div>)}</blockquote>);
      continue;
    }

    if (/^\s*(?:---+|___+|\*\*\*+)\s*$/.test(line)) {
      blocks.push(<hr key={`hr-${index}`} />);
      index += 1;
      continue;
    }

    const listMatch = line.match(/^\s*(?:([-+*•])|(\d+)\.)\s+(.+)$/);
    if (listMatch) {
      const ordered = Boolean(listMatch[2]);
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*(?:([-+*•])|(\d+)\.)\s+(.+)$/);
        if (!item || Boolean(item[2]) !== ordered) break;
        items.push(item[3]);
        index += 1;
      }
      const List = ordered ? "ol" : "ul";
      blocks.push(<List key={`list-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}><InlineText text={item} /></li>)}</List>);
      continue;
    }

    const paragraph = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim() && !/^\s{0,3}#{1,6}\s/.test(lines[index]) && !/^\s*\|/.test(lines[index]) && !/^\s*(?:[-+*•]|\d+\.)\s+/.test(lines[index]) && !/^\s*```/.test(lines[index])) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push(<p key={`p-${index}`}><InlineText text={paragraph.join(" ")} /></p>);
  }
  return <div className="chat-message-content">{blocks}</div>;
}

function Chart({ chart }) {
  if (!chart?.points?.length) return null;
  if (chart.type === "scatter") {
    const values = chart.points;
    const xs = values.map((p) => Number(p.x)); const ys = values.map((p) => Number(p.y));
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    return <div className="chat-chart"><h4>{chart.column} by {chart.group_by}</h4><svg viewBox="0 0 600 220" role="img" aria-label="Scatter plot">{values.map((p, i) => <circle key={i} cx={30 + ((p.x - minX) / (maxX - minX || 1)) * 540} cy={200 - ((p.y - minY) / (maxY - minY || 1)) * 180} r="3" />)}</svg></div>;
  }
  const maximum = Math.max(...chart.points.map((p) => p.value), 1);
  return <div className="chat-chart"><h4>{chart.type === "histogram" ? `${chart.column} distribution` : `${chart.column} counts`}</h4><div className="chat-chart-bars">{chart.points.map((p, i) => <div className="chat-chart-column" key={`${p.label}-${p.series || i}`} title={`${p.label}${p.series ? ` · ${p.series}` : ""}: ${p.value}`}><span>{p.value}</span><i style={{ height: `${Math.max(3, (p.value / maximum) * 100)}%` }} /><small>{p.series ? `${p.label} · ${p.series}` : p.label}</small></div>)}</div></div>;
}

const WELCOME = { role: "bot", text: "Hi! I can explore your data, explain patterns, prepare charts, suggest cleanup, and apply edits with your confirmation." };

export default function Chat({ sessionId, onMessagesChanged }) {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    let active = true;
    if (!sessionId) return () => { active = false; };
    api(`/chat/sessions/${encodeURIComponent(sessionId)}`).then((data) => {
      if (active) setMessages(data.messages?.length ? data.messages.map((item) => {
        const metadata = JSON.parse(item.metadata || "{}");
        return { role: item.role === "assistant" ? "bot" : item.role, text: item.content, chart: metadata.chart, pending: metadata.confirmation_id ? { id: metadata.confirmation_id, details: metadata.pending_action } : null };
      }) : [WELCOME]);
    }).catch(() => { if (active) setMessages([WELCOME]); });
    return () => { active = false; };
  }, [sessionId]);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function send(text, confirmationId) {
    const message = (text ?? input).trim();
    if ((!message && !confirmationId) || sending || !sessionId) return;
    if (message) setMessages((m) => [...m, { role: "user", text: message }]);
    setInput(""); setSending(true);
    try {
      const history = messages.filter((m) => m.role === "user" || m.role === "bot").slice(-12).map((m) => ({ role: m.role === "bot" ? "assistant" : "user", content: m.text }));
      const res = await api("/chat", { method: "POST", body: { message, history, session_id: sessionId, ...(confirmationId ? { confirm_action: confirmationId } : {}) }, timeout: 180000 });
      if (res.dataset_changed) {
        localStorage.removeItem(`automl-pinned-charts:${sessionId || "default"}`);
        window.dispatchEvent(new CustomEvent("analysis:clear-charts"));
        window.dispatchEvent(new CustomEvent("dataset:changed"));
      }
      if (res.chart) {
        const chartKey = `automl-pinned-charts:${sessionId || "default"}`;
        const pinned = JSON.parse(localStorage.getItem(chartKey) || "[]");
        localStorage.setItem(chartKey, JSON.stringify([res.chart, ...pinned.filter((c) => !(c.type === res.chart.type && c.column === res.chart.column))].slice(0, 8)));
        window.dispatchEvent(new CustomEvent("analysis:pin-chart", { detail: res.chart }));
      }
      setMessages((m) => [...m, { role: "bot", text: res.reply, chart: res.chart, pending: res.confirmation_id ? { id: res.confirmation_id, details: res.pending_action } : null }]);
      onMessagesChanged?.();
    } catch (e) { setMessages((m) => [...m, { role: "bot", text: `⚠ ${e.message}` }]); }
    finally { setSending(false); }
  }

  return <div className="panel chat-box" style={{ padding: 0 }}>
    <div className="chat-log">
      {messages.map((m, i) => <div key={i} className={`msg ${m.role}${m.role === "bot" && /^\s*\|.*\n\s*\|\s*:?-{3,}/m.test(m.text) ? " msg-has-table" : ""}`}>
        <MessageContent text={m.text} />
        {m.chart && <Chart chart={m.chart} />}
        {m.pending && <div className="chat-confirm"><pre>{JSON.stringify(m.pending.details.arguments, null, 2)}</pre><button className="primary-button" onClick={() => send("", m.pending.id)} disabled={sending}>{m.pending.details.operation === "train_models" ? "Confirm model training" : "Confirm dataset edit"}</button><button className="secondary-button" onClick={() => setMessages((old) => old.map((x, n) => n === i ? { ...x, pending: null } : x))} disabled={sending}>Cancel</button></div>}
      </div>)}
      {sending && <div className="msg bot">Analyzing…</div>}
      <div ref={endRef} />
    </div>
    <div className="chips">{SUGGESTIONS.map((s) => <button key={s} className="chip" onClick={() => send(s)} disabled={sending || !sessionId}>{s}</button>)}</div>
    <div className="chat-input"><input value={input} placeholder="Ask a question about your data…" onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} disabled={!sessionId} /><button className="primary-button" onClick={() => send()} disabled={sending || !sessionId || !input.trim()}>Send</button></div>
  </div>;
}
