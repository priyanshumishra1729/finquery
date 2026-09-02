import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  BarChart3,
  BookOpenText,
  CircleDollarSign,
  FileSearch,
  LineChart,
  Loader2,
  MessageSquarePlus,
  PanelLeft,
  ReceiptText,
  Sparkles,
} from "lucide-react";

import { API_BASE_URL, checkBackendHealth, sendChatMessage } from "./services/api";

const suggestedQuestions = [
  "What is the difference between revenue and net income?",
  "What is EBITDA?",
  "How is gross margin calculated?",
  "What does a balance sheet show?",
];

const recentChats = [
  "Revenue vs net income",
  "Gross margin calculation",
  "Balance sheet overview",
];

function formatTime(date) {
  return new Intl.DateTimeFormat([], {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const messagesEndRef = useRef(null);

  const canSubmit = input.trim().length > 0 && !isLoading;

  const chatCount = useMemo(
    () => messages.filter((message) => message.role === "assistant").length,
    [messages],
  );

  useEffect(() => {
    checkBackendHealth().then(setIsBackendConnected);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  async function submitMessage(messageText = input) {
    const trimmedMessage = messageText.trim();
    if (!trimmedMessage || isLoading) {
      return;
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmedMessage,
      timestamp: new Date(),
    };

    setMessages((currentMessages) => [...currentMessages, userMessage]);
    setInput("");
    setError("");
    setIsLoading(true);

    try {
      const data = await sendChatMessage(trimmedMessage);
      setIsBackendConnected(true);
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.response,
          model: data.model,
          timestamp: new Date(),
        },
      ]);
    } catch {
      setIsBackendConnected(false);
      setError("Unable to connect to FinQuery. Please make sure the backend and Ollama are running.");
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitMessage();
    }
  }

  function startNewChat() {
    setMessages([]);
    setInput("");
    setError("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="FinQuery navigation">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">
            <LineChart size={22} />
          </div>
          <div>
            <p className="brand-name">FinQuery</p>
            <p className="brand-subtitle">Financial Assistant</p>
          </div>
        </div>

        <button className="new-chat-button" type="button" onClick={startNewChat}>
          <MessageSquarePlus size={18} />
          <span>New Chat</span>
        </button>

        <div className="sidebar-section">
          <p className="sidebar-label">Recent Chats</p>
          <div className="recent-list">
            {recentChats.map((chat) => (
              <button className="recent-item" key={chat} type="button" onClick={() => setInput(chat)}>
                <ReceiptText size={16} />
                <span>{chat}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="connection-card">
            <span className={isBackendConnected ? "status-dot connected" : "status-dot"} />
            <div>
              <p className="connection-title">
                {isBackendConnected ? "AI backend connected" : "Backend not connected"}
              </p>
              <p className="connection-url">{API_BASE_URL}</p>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button mobile-only" type="button" aria-label="Open navigation">
            <PanelLeft size={19} />
          </button>
          <div>
            <h1>FinQuery</h1>
            <p>Financial Document Intelligence Assistant</p>
          </div>
          <div className="model-chip" aria-label="Current model">
            <Sparkles size={15} />
            <span>codellama</span>
          </div>
        </header>

        <section className="workspace" aria-label="Chat workspace">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon" aria-hidden="true">
                <FileSearch size={30} />
              </div>
              <p className="eyebrow">AI financial assistant</p>
              <h2>Welcome to FinQuery</h2>
              <p className="welcome-copy">
                Ask questions about financial concepts, accounting, valuation, and financial analysis.
              </p>
              <p className="exercise-note">
                FinQuery is currently powered by Code Llama through the local FastAPI backend.
                It provides informational guidance and does not claim access to documents that were not provided.
              </p>

              <div className="suggestion-grid" aria-label="Suggested questions">
                {suggestedQuestions.map((question) => (
                  <button
                    className="suggestion-card"
                    key={question}
                    type="button"
                    onClick={() => submitMessage(question)}
                    disabled={isLoading}
                  >
                    <BookOpenText size={18} />
                    <span>{question}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list" aria-live="polite">
              {messages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  <div className="message-avatar" aria-hidden="true">
                    {message.role === "assistant" ? <CircleDollarSign size={18} /> : <BarChart3 size={18} />}
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      <span>{message.role === "assistant" ? "FinQuery" : "You"}</span>
                      <span>{formatTime(message.timestamp)}</span>
                      {message.model ? <span>{message.model}</span> : null}
                    </div>
                    <p>{message.content}</p>
                  </div>
                </article>
              ))}

              {isLoading ? (
                <article className="message assistant loading-message">
                  <div className="message-avatar" aria-hidden="true">
                    <Loader2 className="spin" size={18} />
                  </div>
                  <div className="message-body">
                    <div className="message-meta">
                      <span>FinQuery</span>
                    </div>
                    <p>FinQuery is thinking...</p>
                  </div>
                </article>
              ) : null}

              <div ref={messagesEndRef} />
            </div>
          )}

          {error ? (
            <div className="error-banner" role="alert">
              {error}
            </div>
          ) : null}
        </section>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            submitMessage();
          }}
        >
          <label className="sr-only" htmlFor="chat-input">
            Ask FinQuery a financial question
          </label>
          <textarea
            id="chat-input"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about revenue, margins, balance sheets, valuation..."
            rows={1}
          />
          <button type="submit" disabled={!canSubmit} aria-label="Send message">
            {isLoading ? <Loader2 className="spin" size={18} /> : <ArrowUp size={18} />}
          </button>
        </form>
      </main>

      <aside className="insight-rail" aria-label="Financial context examples">
        <p className="rail-label">Financial Signals</p>
        {[
          ["Revenue", "$7.8M", "+12%"],
          ["Gross Margin", "42%", "+3%"],
          ["Operating Expense", "$4.8M", "-5%"],
          ["Cash Runway", "18 mo", "stable"],
        ].map(([label, value, change]) => (
          <div className="metric-card" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{change}</small>
          </div>
        ))}
        <p className="rail-note">
          Sample indicators for visual context only. Live document analysis begins in later exercises.
        </p>
      </aside>
    </div>
  );
}

export default App;
