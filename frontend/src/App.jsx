import { useEffect, useRef, useState } from "react";
import "./App.css";
import "./pages.css";
import "./neumorphic.css";
import { api } from "./api";
import Dashboard from "./pages/Dashboard";
import Dataset from "./pages/Dataset";
import Analysis from "./pages/Analysis";
import Models from "./pages/Models";
import Prediction from "./pages/Prediction";
import Chat from "./pages/Chat";

const TABS = [
  { id: "dashboard", label: "Dashboard", title: "Dashboard", subtitle: "Overview of your AutoML workspace" },
  { id: "dataset", label: "Dataset", title: "Dataset Manager", subtitle: "Upload and process your CSV dataset." },
  { id: "analysis", label: "Analysis", title: "Data Analysis", subtitle: "Statistics and charts for your dataset" },
  { id: "models", label: "Models", title: "Models", subtitle: "View the saved model and training comparison" },
  { id: "prediction", label: "Prediction", title: "Prediction", subtitle: "Run the trained model on new data" },
  { id: "chat", label: "AI Chat", title: "AI Assistant", subtitle: "Ask questions about your dataset" },
];

export default function App() {
  const [active, setActive] = useState("dashboard");
  const [online, setOnline] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(() => localStorage.getItem("automl-chat-session"));
  const [profileOpen, setProfileOpen] = useState(false);
  const [editingSession, setEditingSession] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [historyError, setHistoryError] = useState("");
  const didLoadSessions = useRef(false);

  const refreshSessions = async () => {
    try { const data = await api("/chat/sessions"); setSessions(data.sessions || []); }
    catch { /* the backend status indicator reports the connection state */ }
  };

  const selectSession = (id) => {
    setSessionId(id);
    localStorage.setItem("automl-chat-session", id);
  };

  useEffect(() => {
    if (didLoadSessions.current) return;
    didLoadSessions.current = true;
    api("/chat/sessions").then(async (data) => {
      setSessions(data.sessions || []);
      if (data.sessions?.length) selectSession(data.sessions[0].id);
      else {
        const created = await api("/chat/sessions", { method: "POST", body: {} });
        selectSession(created.session.id);
        refreshSessions();
      }
    }).catch(() => {});
  }, []);

  async function newChat() {
    try {
      const data = await api("/chat/sessions", { method: "POST", body: {} });
      selectSession(data.session.id);
      setActive("dashboard");
      refreshSessions();
    } catch { setActive("dashboard"); }
  }

  async function renameChat(id) {
    const title = renameValue.trim();
    if (!title) { setHistoryError("Enter a chat name."); return; }
    try {
      await api(`/chat/sessions/${encodeURIComponent(id)}`, { method: "PATCH", body: { title } });
      setEditingSession(null);
      setHistoryError("");
      await refreshSessions();
    } catch (error) { setHistoryError(error.message); }
  }

  // Ping the backend so the sidebar shows the real connection state.
  useEffect(() => {
    let alive = true;
    const ping = () =>
      api("/health", { timeout: 4000 })
        .then(() => alive && setOnline(true))
        .catch(() => alive && setOnline(false));
    ping();
    const id = setInterval(ping, 8000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const tab = TABS.find((t) => t.id === active);

  const pages = {
    dashboard: <Dashboard onNavigate={setActive} />,
    dataset: <Dataset />,
    analysis: <Analysis sessionId={sessionId} onNavigate={setActive} />,
    models: <Models />,
    prediction: <Prediction onNavigate={setActive} />,
    chat: <Chat key={sessionId || "chat"} sessionId={sessionId} onMessagesChanged={refreshSessions} />,
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-chat-heading">
          <div className="logo-icon">AI</div><strong>Workspace</strong>
        </div>
        <button className="new-chat-button" onClick={newChat}><span>＋</span> New chat</button>
        <div className="history-title">Recent chats</div>
        <nav className="chat-history">
          {historyError && <small className="history-error">{historyError}</small>}
          {sessions.map((session) => <div key={session.id} className={`history-row ${session.id === sessionId ? "selected" : ""}`}>
            {editingSession === session.id ? <div className="history-rename">
              <input autoFocus value={renameValue} aria-label="Chat name" onChange={(event) => setRenameValue(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") renameChat(session.id); if (event.key === "Escape") setEditingSession(null); }} />
              <button onClick={() => renameChat(session.id)} aria-label="Save chat name" title="Save">✓</button>
              <button onClick={() => setEditingSession(null)} aria-label="Cancel rename" title="Cancel">×</button>
            </div> : <>
              <button className="history-item" onClick={() => { selectSession(session.id); setActive("dashboard"); }} title={session.title}>{session.title}</button>
              <button className="history-rename-trigger" onClick={() => { setEditingSession(session.id); setRenameValue(session.title); setHistoryError(""); }} aria-label={`Rename ${session.title}`} title="Rename chat">✎</button>
            </>}
          </div>)}
          {!sessions.length && <small className="history-empty">Your conversations will appear here.</small>}
        </nav>
        <div className="sidebar-bottom">
          <div className="backend-status">
            <span className="status-dot" style={online ? undefined : { background: "#ff6b6b", boxShadow: "0 0 12px rgba(255,107,107,.7)" }} />
            <strong>{online ? "Backend online" : "Backend offline"}</strong>
          </div>
          <button className="profile-button" onClick={() => setProfileOpen((value) => !value)} aria-expanded={profileOpen}>
            <span className="profile">S</span><span className="profile-label"><strong>Shiva</strong><small>Local workspace</small></span><span className="profile-more">···</span>
          </button>
          {profileOpen && <div className="profile-menu"><span>SQLite storage</span><small>Local app data</small></div>}
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="brand-title"><h1>AutoML</h1><span>LLM Framework</span></div>
          <nav className="top-navigation" aria-label="Main navigation">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`nav-button ${active === t.id ? "active" : ""}`}
              onClick={() => setActive(t.id)}
            >
              {t.label}
            </button>
          ))}
          </nav>
          <div className="top-actions">
            <div className="connection">
              <span />
              {online ? "Connected" : "Disconnected"}
            </div>
          </div>
        </header>

        <div className="page-heading"><h2>{tab.title}</h2><p>{tab.subtitle}</p></div>

        <main className="content">
          <div className="page-wrap">{pages[active]}</div>
        </main>
      </div>
    </div>
  );
}
