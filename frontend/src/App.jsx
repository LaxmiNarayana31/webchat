import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Menu,
  X,
  Globe,
  AlertCircle,
  RefreshCcw,
  Plus,
  MessageSquare,
  ShieldCheck,
  Cpu,
  Clock,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  ExternalLink,
  Copy,
  Check,
  Trash2,
  Search,
  Sparkles,
  Layers,
  FileText,
  ArrowRight,
  BookOpen,
  ThumbsUp,
  ThumbsDown,
  Zap,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://localhost:8000/api";

function getOrCreateClientId() {
  let cid = localStorage.getItem("webchat_client_id");
  if (!cid) {
    cid =
      "client_" +
      (window.crypto?.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2) + Date.now().toString(36));
    localStorage.setItem("webchat_client_id", cid);
  }
  return cid;
}

// URL Validation: checks for valid domain format (with >= 2-letter TLD), localhost, or IPv4
const DOMAIN_REGEX =
  /^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?::\d+)?(?:\/.*)?$/;
const IP_OR_LOCAL_REGEX = /^(?:localhost|127\.0\.0\.1)(?::\d+)?(?:\/.*)?$/;

function isValidUrlPattern(str) {
  if (!str || typeof str !== "string") return false;
  const trimmed = str.trim();
  if (trimmed.length < 3 || /\s/.test(trimmed)) return false;

  // Strip protocol prefix if present
  const withoutProtocol = trimmed.replace(/^https?:\/\//i, "");
  if (!withoutProtocol) return false;

  // Check against strict domain or localhost pattern
  if (
    !DOMAIN_REGEX.test(withoutProtocol) &&
    !IP_OR_LOCAL_REGEX.test(withoutProtocol)
  ) {
    return false;
  }

  // Verify valid URL constructor parsing
  try {
    const full = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
    const parsed = new URL(full);
    return (
      (parsed.protocol === "http:" || parsed.protocol === "https:") &&
      !!parsed.hostname
    );
  } catch {
    return false;
  }
}

function parseUrlInput(text) {
  if (!text) return { valid: [], invalid: [] };
  const rawItems = text
    .split(/[\r\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const valid = [];
  const invalid = [];

  for (const item of rawItems) {
    if (isValidUrlPattern(item)) {
      let u = item;
      if (!u.startsWith("http://") && !u.startsWith("https://")) {
        u = "https://" + u;
      }
      if (!valid.includes(u)) {
        valid.push(u);
      }
    } else {
      if (!invalid.includes(item)) {
        invalid.push(item);
      }
    }
  }

  return { valid, invalid };
}

function formatModelName(name) {
  if (!name) return "Gemini 3.6 Flash";
  const str = String(name).trim();
  if (/gemini-3\.6/i.test(str)) return "Gemini 3.6 Flash";
  if (/gemini-2\.0/i.test(str)) return "Gemini 2.0 Flash";
  if (/gemini-1\.5/i.test(str)) return "Gemini 1.5 Pro";
  if (/llama-3\.3/i.test(str)) return "Llama 3.3 70B";
  if (/deepseek/i.test(str)) return "DeepSeek R1 70B";
  return str.replace(/-/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
}

function getSuggestedQuestions(activeDoc, messages = []) {
  const rawTitle = (activeDoc?.title || activeDoc?.url || "").trim();
  const cleanTitle =
    rawTitle
      .replace(/^Domain Crawl \(\d+ pages\)/i, "")
      .replace(/^Site Crawl:/i, "")
      .trim() || "extracted content";
  const shortTitle =
    cleanTitle.length > 32 ? cleanTitle.substring(0, 30) + "..." : cleanTitle;

  // Dynamic context-aware follow-up questions during active chat
  if (messages && messages.length > 0) {
    const userMsgs = messages.filter((m) => m.role === "user");
    const assistantMsgs = messages.filter((m) => m.role === "assistant");
    const lastUserQuery =
      userMsgs.length > 0 ? userMsgs[userMsgs.length - 1].content || "" : "";
    const lastAssistant =
      assistantMsgs.length > 0 ? assistantMsgs[assistantMsgs.length - 1] : null;
    const lastContent = lastAssistant?.content || "";
    const citations = lastAssistant?.citations || [];

    const combinedText = (lastUserQuery + " " + lastContent).toLowerCase();

    // 1. Follow-up based on grounded source citations in last response
    if (citations && citations.length > 0) {
      const topCitationTitle = citations[0].title
        ? citations[0].title.substring(0, 28) + "..."
        : "the cited sources";
      return [
        `🔍 Tell me more about "${topCitationTitle}" from the sources`,
        `📌 What additional evidence or details are in the cited sources?`,
        `⚡ What are the key practical implications from these sources?`,
      ];
    }

    // 2. Follow-up based on Code / API / Implementation context
    if (
      combinedText.includes("code") ||
      combinedText.includes("python") ||
      combinedText.includes("function") ||
      combinedText.includes("api") ||
      combinedText.includes("script") ||
      combinedText.includes("example")
    ) {
      return [
        `🔍 Show a step-by-step code implementation or script`,
        `⚡ What are the main edge cases or error handling steps?`,
        `📌 How can this code be customized or extended further?`,
      ];
    }

    // 3. Follow-up based on RAG / Vector / Architecture / Performance
    if (
      combinedText.includes("rag") ||
      combinedText.includes("vector") ||
      combinedText.includes("retrieval") ||
      combinedText.includes("benchmark") ||
      combinedText.includes("performance") ||
      combinedText.includes("embedding")
    ) {
      return [
        `🔍 What are the performance trade-offs and latency metrics?`,
        `📌 How does vector search compare with keyword retrieval here?`,
        `⚡ What optimization techniques are recommended?`,
      ];
    }

    // 4. Follow-up based on AI / LLM / Multi-Agent models
    if (
      combinedText.includes("ai") ||
      combinedText.includes("llm") ||
      combinedText.includes("model") ||
      combinedText.includes("agent")
    ) {
      return [
        `🔍 How does this model or pipeline compare with alternatives?`,
        `📌 What are the core limitations or trade-offs mentioned?`,
        `⚡ What future improvements or benchmarks are highlighted?`,
      ];
    }

    // 5. General follow-up based on last user question
    if (lastUserQuery) {
      const shortQuery =
        lastUserQuery.length > 30
          ? lastUserQuery.substring(0, 28) + "..."
          : lastUserQuery;
      return [
        `📌 Can you elaborate further on "${shortQuery}"?`,
        `🔍 What are the practical real-world applications of this?`,
        `⚡ Summarize the key action items or conclusions`,
      ];
    }
  }

  // Default suggested questions prior to sending any messages
  if (!activeDoc) {
    return [
      "📌 Summarize core takeaways",
      "🔍 Extract key technical architecture",
      "📊 What are the limitations or trade-offs?",
    ];
  }

  const titleLower = cleanTitle.toLowerCase();

  if (
    titleLower.includes("rag") ||
    titleLower.includes("retrieval") ||
    titleLower.includes("vector")
  ) {
    return [
      `📌 What are the core RAG components in "${shortTitle}"?`,
      `🔍 How does vector search & retrieval work in this article?`,
      `⚡ What are the main architectural trade-offs?`,
    ];
  }

  if (
    titleLower.includes("python") ||
    titleLower.includes("code") ||
    titleLower.includes("api") ||
    titleLower.includes("docs")
  ) {
    return [
      `📌 What are the main features explained in "${shortTitle}"?`,
      `🔍 Give a code example demonstrating core functions`,
      `⚡ What are the key API guidelines or best practices?`,
    ];
  }

  if (
    titleLower.includes("ai") ||
    titleLower.includes("model") ||
    titleLower.includes("llm") ||
    titleLower.includes("agent")
  ) {
    return [
      `📌 What are the primary AI insights in "${shortTitle}"?`,
      `🔍 How does the agentic pipeline or model structure operate?`,
      `📊 What are the key benchmarks and findings?`,
    ];
  }

  return [
    `📌 Summarize key takeaways from "${shortTitle}"`,
    `🔍 What are the primary technical concepts explained?`,
    `⚡ What are the most important conclusions or recommendations?`,
  ];
}

export default function App() {
  // Client & Quota State
  const [clientId] = useState(getOrCreateClientId);
  const [quota, setQuota] = useState(null);

  // Navigation & Modals
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showDocModal, setShowDocModal] = useState(false);
  const [showModelModal, setShowModelModal] = useState(false);

  // Sessions
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [sessionSearch, setSessionSearch] = useState("");

  // Ingestion State
  const [urlsInput, setUrlsInput] = useState("");
  const [strategy, setStrategy] = useState("auto");
  const [enableCrawl, setEnableCrawl] = useState(false);
  const [crawlMaxPages, setCrawlMaxPages] = useState(5);
  const [crawlMaxDepth, setCrawlMaxDepth] = useState(2);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState("");

  // Active Document Context
  const [activeDoc, setActiveDoc] = useState(() => {
    try {
      const saved = localStorage.getItem("webchat_active_doc");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  // Keep webchat_active_doc in sync with localStorage
  useEffect(() => {
    if (activeDoc) {
      try {
        const docSummary = {
          url: activeDoc.url,
          title: activeDoc.title,
          strategyUsed: activeDoc.strategyUsed,
          wordCount: activeDoc.wordCount,
          paywallBypassed: activeDoc.paywallBypassed,
          pagesCrawled: activeDoc.pagesCrawled,
          content:
            activeDoc.content && activeDoc.content.length < 50000
              ? activeDoc.content
              : "",
        };
        localStorage.setItem("webchat_active_doc", JSON.stringify(docSummary));
      } catch (e) {
        console.debug("Could not cache activeDoc in localStorage", e);
      }
    } else {
      localStorage.removeItem("webchat_active_doc");
    }
  }, [activeDoc]);

  // Chat State
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [expandedThinking, setExpandedThinking] = useState({});
  const [copiedCodeIdx, setCopiedCodeIdx] = useState(null);
  const [feedback, setFeedback] = useState({});

  // Model Catalog
  const [models, setModels] = useState([]);

  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const isNearBottom = useRef(true);
  const { valid: validUrls, invalid: invalidUrls } = parseUrlInput(urlsInput);

  // Track scroll position to avoid fighting user during streaming
  const handleChatScroll = () => {
    const el = chatContainerRef.current;
    if (!el) return;
    isNearBottom.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 150;
  };

  // Initial Data Fetching
  useEffect(() => {
    fetchQuota();
    fetchModels();
    fetchSessions();
    const savedSessionId = localStorage.getItem("webchat_active_session_id");
    if (savedSessionId) {
      loadSession(savedSessionId);
    }
  }, []);

  // Smart auto-scroll: only scroll down if user is already near the bottom
  useEffect(() => {
    if (isNearBottom.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Fetch Functions
  const fetchSessions = async () => {
    if (!clientId) return;
    try {
      const params = new URLSearchParams({ client_id: clientId });
      const res = await fetch(`${API_BASE}/user/sessions?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setSessions(Array.isArray(data) ? data : []);
      }
    } catch (e) {
      console.debug("Sessions fetch error", e);
    }
  };

  // Fetch Functions
  const fetchQuota = async () => {
    try {
      const params = new URLSearchParams({ client_id: clientId });
      const res = await fetch(`${API_BASE}/user/quota?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setQuota(data);
      }
    } catch (e) {
      console.debug("Quota fetch error", e);
    }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/models`);
      if (res.ok) {
        const data = await res.json();
        setModels(Array.isArray(data) ? data : []);
      }
    } catch (e) {
      console.debug("Models fetch error", e);
    }
  };

  // Ingestion Handler
  const handleIngest = async () => {
    if (validUrls.length === 0) {
      alert(
        "Please enter at least one valid website URL (e.g. example.com or https://docs.python.org).",
      );
      return;
    }

    if (invalidUrls.length > 0) {
      alert(
        `Invalid URL format detected: "${invalidUrls.join('", "')}". Please enter a valid website domain or remove invalid entries.`,
      );
      return;
    }

    setIsIngesting(true);
    setIngestStatus(
      enableCrawl
        ? `Initiating domain crawl on ${validUrls[0]}...`
        : `Extracting ${validUrls.length} URL(s)...`,
    );

    try {
      let res, data;
      if (enableCrawl && validUrls.length > 0) {
        res = await fetch(`${API_BASE}/crawl`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: validUrls[0],
            max_pages: crawlMaxPages,
            max_depth: crawlMaxDepth,
            strategy: strategy,
          }),
        });
        data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Domain crawl failed");

        setActiveDoc({
          url: data.root_url,
          title: `Domain Crawl (${data.pages_crawled} pages)`,
          content: `Crawled ${data.pages_crawled} sub-pages totaling ${data.total_words} words.`,
          wordCount: data.total_words,
          strategyUsed: strategy,
          paywallBypassed: false,
          pagesCrawled: data.pages_crawled,
        });
      } else {
        res = await fetch(`${API_BASE}/scrape`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: validUrls.join(", "),
            urls: validUrls,
            strategy: strategy,
          }),
        });
        data = await res.json();
        if (!res.ok)
          throw new Error(
            data.detail?.message || data.detail || "Extraction failed",
          );

        setActiveDoc({
          url: data.url,
          title: data.title || validUrls[0],
          content: data.content,
          wordCount: data.word_count,
          strategyUsed: data.strategy_used,
          paywallBypassed: data.paywall_bypassed,
          pagesCrawled: data.pages_crawled,
        });
      }

      // Reset current session ID so a fresh session is lazily created on the first chat prompt
      setCurrentSessionId(null);
      localStorage.removeItem("webchat_active_session_id");

      // Transition to Chat View cleanly without fake system message
      setMessages([]);
    } catch (err) {
      alert("Extraction Error: " + err.message);
    } finally {
      setIsIngesting(false);
      setIngestStatus("");
    }
  };

  // Send Message
  const handleSendMessage = async (customQuery = null) => {
    const query = (customQuery || inputMessage).trim();
    if (!query || isGenerating) return;

    if (quota && !quota.can_request) {
      alert("Daily query limit reached. Quota resets at midnight UTC.");
      return;
    }

    if (!customQuery) setInputMessage("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setIsGenerating(true);

    // Placeholder for assistant turn
    const assistantIndex = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "",
        modelInfo: "Agent Orchestrator",
        provider: "",
        latency: 0,
        isThinking: true,
        steps: [],
        citations: [],
      },
    ]);

    try {
      const payload = {
        query: query,
        url: activeDoc?.url || validUrls.join(", ") || null,
        document_content: activeDoc?.content || null,
        stream: isStreaming,
        session_id: currentSessionId,
        client_id: clientId,
      };

      if (isStreaming) {
        const response = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(
            err.detail?.message || err.detail || "Chat request failed",
          );
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = "";
        let buffer = "";
        let currentEvent = "message";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();

          for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line) continue;

            if (line.startsWith("event: ")) {
              currentEvent = line.substring(7).trim();
            } else if (line.startsWith("data: ")) {
              const dataStr = line.substring(6).trim();
              if (dataStr === "[DONE]") break;

              try {
                const parsed = JSON.parse(dataStr);
                if (currentEvent === "session") {
                  if (parsed.session_id) {
                    setCurrentSessionId(parsed.session_id);
                    localStorage.setItem(
                      "webchat_active_session_id",
                      parsed.session_id,
                    );
                  }
                  if (parsed.quota) {
                    setQuota(parsed.quota);
                  }
                }

                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (!last || last.role !== "assistant") return prev;

                  if (currentEvent === "step") {
                    last.steps = [...(last.steps || []), parsed];
                  } else if (currentEvent === "citations") {
                    last.citations = parsed;
                  } else if (parsed.chunk !== undefined) {
                    last.isThinking = false;
                    fullAnswer += parsed.chunk;
                    last.content = fullAnswer;
                    if (parsed.model_used) last.modelInfo = parsed.model_used;
                    if (parsed.provider) last.provider = parsed.provider;
                    if (parsed.latency_sec) last.latency = parsed.latency_sec;
                  }
                  return updated;
                });
              } catch (e) {
                // Ignore raw chunk parsing
              }
            }
          }
        }
      } else {
        // Batch query
        const res = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok)
          throw new Error(
            data.detail?.message || data.detail || "Chat request failed",
          );

        if (data.session_id) {
          setCurrentSessionId(data.session_id);
          localStorage.setItem("webchat_active_session_id", data.session_id);
        }

        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last) {
            last.isThinking = false;
            last.content = data.answer;
            last.modelInfo = data.model_used;
            last.provider = data.provider;
            last.latency = data.latency_sec;
            last.citations = data.citations || [];
          }
          return updated;
        });
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last) {
          last.isThinking = false;
          last.content = `> ⚠️ **Error:** ${err.message}`;
        }
        return updated;
      });
    } finally {
      setIsGenerating(false);
      fetchQuota();
      fetchSessions();
    }
  };

  // Start New Chat with Current Document (keeps activeDoc intact)
  const startNewChatForCurrentDoc = () => {
    localStorage.removeItem("webchat_active_session_id");
    setCurrentSessionId(null);
    setMessages([]);
    fetchSessions();
  };

  // Start Fresh with New URL (clears activeDoc and returns to URL input screen)
  const startFreshWithNewUrl = () => {
    localStorage.removeItem("webchat_active_session_id");
    localStorage.removeItem("webchat_active_doc");
    setActiveDoc(null);
    setCurrentSessionId(null);
    setMessages([]);
    setUrlsInput("");
    fetchSessions();
  };

  // Alias for backwards compatibility
  const startNewChat = startFreshWithNewUrl;

  // Load Session
  const loadSession = async (sid) => {
    try {
      const res = await fetch(`${API_BASE}/sessions/${sid}`);
      if (res.ok) {
        const data = await res.json();
        setCurrentSessionId(sid);
        localStorage.setItem("webchat_active_session_id", sid);
        if (data.url) {
          setActiveDoc({
            url: data.url,
            title: data.title || data.url,
            content: "",
            wordCount: 0,
            strategyUsed: "Session",
            paywallBypassed: false,
          });
        }
        if (Array.isArray(data.messages)) {
          setMessages(
            data.messages.map((m) => ({
              role: m.role,
              content: m.content,
              citations: m.citations || [],
              modelInfo: m.model_used || "Cached",
              provider: m.provider || "",
              steps: [],
            })),
          );
        }
      }
    } catch (e) {
      console.error("Failed to load session", e);
    }
  };

  // Delete Session
  const deleteSession = async (e, sid) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this session?")) return;
    try {
      await fetch(`${API_BASE}/sessions/${sid}`, { method: "DELETE" });
      if (currentSessionId === sid) {
        localStorage.removeItem("webchat_active_session_id");
        startNewChat();
      } else {
        fetchSessions();
      }
    } catch (err) {
      alert("Failed to delete session: " + err.message);
    }
  };

  // Copy Code Handler
  const copyCode = (code, idx) => {
    navigator.clipboard.writeText(code);
    setCopiedCodeIdx(idx);
    setTimeout(() => setCopiedCodeIdx(null), 2000);
  };

  // Filtered Sessions
  const filteredSessions = sessions.filter((s) =>
    (s.title || s.url || "")
      .toLowerCase()
      .includes(sessionSearch.toLowerCase()),
  );

  const totalQuota = quota?.daily_limit || 50;
  const remainingQuota = quota?.requests_remaining ?? quota?.remaining ?? 50;

  return (
    <div className="flex h-screen w-full bg-[#08090c] text-slate-200 overflow-hidden font-sans">
      {/* Background Aurora Ambient Glow */}
      <div className="aurora-bg">
        <div className="aurora-glow-1" />
        <div className="aurora-glow-2" />
      </div>

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 290, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="h-full border-r border-white/10 bg-[#0d0f14]/90 backdrop-blur-xl flex flex-col shrink-0 z-30">
            {/* Sidebar Brand Header */}
            <div className="p-4 border-b border-white/5 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/25">
                  <Globe className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h1 className="font-bold text-sm text-white tracking-wide">
                    WebChat AI
                  </h1>
                </div>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1.5 hover:bg-white/10 rounded-md text-slate-400 hover:text-white transition-colors md:hidden">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* New Session Action Buttons */}
            <div className="p-3 space-y-2">
              {activeDoc ? (
                <>
                  <button
                    onClick={startNewChatForCurrentDoc}
                    className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white py-2.5 px-3 rounded-lg font-medium text-xs shadow-md shadow-sky-500/20 transition-all active:scale-[0.99]"
                    title="Start fresh conversation on the current document">
                    <Plus className="w-4 h-4" /> New Chat (Same Doc)
                  </button>
                  <button
                    onClick={startFreshWithNewUrl}
                    className="w-full flex items-center justify-center gap-1.5 bg-white/5 hover:bg-white/10 text-slate-300 hover:text-white py-2 px-3 rounded-lg font-medium text-xs border border-white/10 transition-all active:scale-[0.99]"
                    title="Clear current document and ingest a new URL">
                    <Globe className="w-3.5 h-3.5 text-sky-400" /> Ingest New
                    URL
                  </button>
                </>
              ) : (
                <button
                  onClick={startFreshWithNewUrl}
                  className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white py-2.5 px-3 rounded-lg font-medium text-xs shadow-md shadow-sky-500/20 transition-all active:scale-[0.99]">
                  <Plus className="w-4 h-4" /> Start New Session
                </button>
              )}
            </div>

            {/* Session Search */}
            <div className="px-3 pb-2">
              <div className="flex items-center gap-2 bg-white/5 border border-white/10 px-2.5 py-1.5 rounded-md text-xs text-slate-400 focus-within:border-sky-500/50">
                <Search className="w-3.5 h-3.5" />
                <input
                  type="text"
                  placeholder="Filter sessions..."
                  value={sessionSearch}
                  onChange={(e) => setSessionSearch(e.target.value)}
                  className="bg-transparent border-none outline-none text-slate-200 text-xs w-full placeholder:text-slate-500"
                />
              </div>
            </div>

            {/* Sessions List */}
            <div className="flex-1 overflow-y-auto px-3 py-1 space-y-1">
              <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider px-2 py-1">
                Chat History ({filteredSessions.length})
              </div>
              {filteredSessions.length === 0 ? (
                <div className="text-xs text-slate-500 text-center py-8 italic">
                  No conversation sessions found
                </div>
              ) : (
                filteredSessions.map((s) => (
                  <div
                    key={s.session_id}
                    onClick={() => loadSession(s.session_id)}
                    className={`group flex items-center justify-between p-2 rounded-lg text-xs cursor-pointer transition-all ${
                      currentSessionId === s.session_id
                        ? "bg-sky-500/15 border border-sky-500/30 text-sky-200 font-medium"
                        : "hover:bg-white/5 text-slate-400 hover:text-slate-200"
                    }`}>
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <MessageSquare className="w-3.5 h-3.5 shrink-0 text-slate-400 group-hover:text-sky-400" />
                      <span className="truncate">
                        {s.title || s.url || "Untitled Chat"}
                      </span>
                    </div>
                    <button
                      onClick={(e) => deleteSession(e, s.session_id)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:text-rose-400 text-slate-500 rounded transition-opacity"
                      title="Delete session">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))
              )}
            </div>

            {/* Device Quota Footer */}
            <div className="p-3 border-t border-white/5 bg-black/20">
              <div className="p-2.5 rounded-xl border border-white/10 bg-white/[0.02]">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Zap className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-[11px] text-slate-400 font-medium">
                      Queries Today
                    </span>
                  </div>
                  <span className="text-[11px] text-emerald-400 font-semibold tabular-nums">
                    {remainingQuota} / {totalQuota}
                  </span>
                </div>
                <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-400 to-emerald-400 transition-all duration-500 rounded-full"
                    style={{
                      width: `${Math.min(100, (remainingQuota / totalQuota) * 100)}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative z-10">
        {/* Top Navbar */}
        <header className="h-14 border-b border-white/10 bg-[#0c0e14]/80 backdrop-blur-md flex items-center justify-between px-4 shrink-0 gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1.5 hover:bg-white/10 rounded-md text-slate-400 hover:text-white transition-colors shrink-0"
              title="Toggle Sidebar">
              <Menu className="w-4 h-4" />
            </button>

            {activeDoc && (
              <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-sky-500/10 border border-sky-500/20 text-xs text-sky-200 min-w-0 max-w-xs md:max-w-md">
                <Globe className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                <span className="truncate font-medium">
                  {activeDoc.title || activeDoc.url}
                </span>
              </div>
            )}
          </div>

          {/* Right Header Space / Actions */}
          <div className="flex items-center gap-2 shrink-0">
            {activeDoc && (
              <>
                <button
                  onClick={startNewChatForCurrentDoc}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-500/15 hover:bg-sky-500/25 border border-sky-500/30 text-sky-300 hover:text-white text-xs font-medium transition-all shadow-sm active:scale-95"
                  title="Start fresh conversation on the current document">
                  <RefreshCcw className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">New Chat</span>
                </button>
                <button
                  onClick={startFreshWithNewUrl}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-slate-300 hover:text-white text-xs font-medium transition-all shadow-sm active:scale-95"
                  title="Clear document and ingest a new URL">
                  <Plus className="w-3.5 h-3.5 text-slate-400" />
                  <span className="hidden sm:inline">Change URL</span>
                </button>
              </>
            )}
          </div>
        </header>

        {/* Content Stream / Home Screen */}
        <main className="flex-1 relative flex flex-col overflow-hidden">
          {messages.length === 0 && !activeDoc ? (
            /* Home Start Screen */
            <div className="flex-1 flex flex-col items-center justify-start pt-6 md:pt-10 p-4 md:p-8 max-w-3xl mx-auto w-full">
              {/* Hero Banner */}
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center mb-4">
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-[11px] font-medium mb-2">
                  <Sparkles className="w-3.5 h-3.5" /> Autonomous Web
                  Intelligence & Agentic RAG
                </div>
                <h2 className="text-xl md:text-2xl font-bold text-white tracking-tight mb-1.5">
                  Chat with Any Web Source
                </h2>
                <p className="text-slate-400 text-xs max-w-lg mx-auto">
                  Extract articles, bypass paywalls, crawl documentation, and
                  ground your questions with multi-model validation.
                </p>
              </motion.div>

              {/* Multi-URL Ingestion Card with Premium Shadow */}
              <motion.div
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className="w-full premium-card-shadow rounded-2xl p-5 border border-white/10">
                {/* Header */}
                <div className="flex items-center justify-between pb-3 mb-3 border-b border-white/10">
                  <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
                    <Globe className="w-4 h-4 text-sky-400" /> Target Website
                    URL(s)
                  </div>
                  <div className="flex items-center gap-2">
                    {urlsInput && (
                      <button
                        onClick={() => setUrlsInput("")}
                        className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors">
                        Clear
                      </button>
                    )}
                    {invalidUrls.length > 0 && (
                      <span className="text-[11px] px-2.5 py-0.5 rounded-full font-medium bg-rose-500/15 text-rose-300 border border-rose-500/30 flex items-center gap-1">
                        <AlertCircle className="w-3 h-3 text-rose-400" />{" "}
                        {invalidUrls.length} Invalid
                      </span>
                    )}
                    <span
                      className={`text-[11px] px-2.5 py-0.5 rounded-full font-medium flex items-center gap-1 ${
                        validUrls.length > 0
                          ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                          : "bg-white/5 text-slate-500"
                      }`}>
                      {validUrls.length > 0 && (
                        <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                      )}
                      {validUrls.length} valid URL
                      {validUrls.length !== 1 ? "s" : ""}
                    </span>
                  </div>
                </div>

                {/* Textarea */}
                <textarea
                  value={urlsInput}
                  onChange={(e) => setUrlsInput(e.target.value)}
                  placeholder="Paste website URL(s) here (e.g. example.com, https://docs.python.org)..."
                  className={`w-full h-32 bg-black/40 border rounded-xl p-3.5 text-sm text-slate-200 placeholder:text-slate-600 resize-none outline-none transition-colors font-mono ${
                    invalidUrls.length > 0
                      ? "border-rose-500/40 focus:border-rose-500/60"
                      : "border-white/5 focus:border-sky-500/50"
                  }`}
                />

                {/* Invalid URL Format Warning Banner */}
                {invalidUrls.length > 0 && (
                  <div className="mt-2.5 p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
                    <div>
                      <span className="font-semibold">
                        Invalid URL pattern:
                      </span>{" "}
                      "{invalidUrls.slice(0, 3).join('", "')}"{" "}
                      {invalidUrls.length > 3
                        ? `and ${invalidUrls.length - 3} more`
                        : ""}
                      . Enter a valid domain name (e.g.{" "}
                      <code className="text-white bg-rose-950/60 px-1 py-0.5 rounded font-mono">
                        example.com
                      </code>{" "}
                      or{" "}
                      <code className="text-white bg-rose-950/60 px-1 py-0.5 rounded font-mono">
                        https://docs.python.org
                      </code>
                      ).
                    </div>
                  </div>
                )}

                {/* Action Submit */}
                <div className="flex items-center justify-between pt-3 mt-3 border-t border-white/5">
                  <div className="text-[11px] text-slate-500">
                    {invalidUrls.length > 0
                      ? "⚠️ Correct the invalid URLs above to enable ingestion."
                      : ingestStatus ||
                        "Automatic paywall bypass and multi-provider vector indexing."}
                  </div>
                  <button
                    onClick={handleIngest}
                    disabled={
                      isIngesting ||
                      validUrls.length === 0 ||
                      invalidUrls.length > 0
                    }
                    className="bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white font-medium text-xs px-5 py-2.5 rounded-xl shadow-xl shadow-sky-500/25 hover:shadow-sky-500/40 flex items-center gap-2 transition-all active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed">
                    {isIngesting ? (
                      <>
                        <RefreshCcw className="w-4 h-4 animate-spin" />{" "}
                        Ingesting & Indexing...
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4" /> Ingest &amp; Start Chat
                      </>
                    )}
                  </button>
                </div>
              </motion.div>

              {/* Feature Highlights Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full mt-6">
                {[
                  {
                    icon: <Layers className="w-4 h-4 text-sky-400" />,
                    title: "Agentic CRAG",
                    desc: "Query rewriting, web fallback, and groundedness checks with LangGraph.",
                  },
                  {
                    icon: <ShieldCheck className="w-4 h-4 text-emerald-400" />,
                    title: "Paywall Bypass",
                    desc: "Apollo State extraction, Jina headless reading, and Wayback Machine fallback.",
                  },
                  {
                    icon: <Cpu className="w-4 h-4 text-cyan-400" />,
                    title: "10-Tier Failover",
                    desc: "Gemini 3.6 Flash auto-cascading to Groq LLaMA models with circuit breakers.",
                  },
                ].map((feat, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-white/[0.02] border border-white/5">
                    <div className="flex items-center gap-2 font-semibold text-xs text-slate-200 mb-1">
                      {feat.icon} {feat.title}
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      {feat.desc}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            /* Active Chat Stream View */
            <div
              ref={chatContainerRef}
              onScroll={handleChatScroll}
              className="flex-1 overflow-y-auto w-full scroll-smooth">
              <div className="p-4 md:p-6 pb-44 md:pb-48 space-y-5 max-w-4xl mx-auto w-full">
                {/* Context Summary Header */}
                {activeDoc && (
                  <div className="p-3 rounded-xl bg-sky-950/20 border border-sky-500/20 flex items-center justify-between text-xs text-slate-300 gap-3">
                    <div className="flex items-center gap-2 min-w-0">
                      <Globe className="w-4 h-4 text-sky-400 shrink-0" />
                      <span className="truncate">
                        Indexed source:{" "}
                        <strong className="text-white">
                          {activeDoc.title || activeDoc.url}
                        </strong>
                      </span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={startNewChatForCurrentDoc}
                        className="px-2.5 py-1 rounded-md bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 hover:text-white font-medium text-[11px] transition-colors flex items-center gap-1.5"
                        title="Start a fresh chat with this document">
                        <RefreshCcw className="w-3 h-3" />
                        <span>New Chat</span>
                      </button>
                      <button
                        onClick={startFreshWithNewUrl}
                        className="px-2.5 py-1 rounded-md bg-white/5 hover:bg-white/10 text-slate-400 hover:text-slate-200 text-[11px] transition-colors"
                        title="Clear document and ingest a new URL">
                        Change URL
                      </button>
                    </div>
                  </div>
                )}

                {/* Centered Quick Start Cards when document is ready but no message sent yet */}
                {messages.length === 0 && activeDoc && (
                  <div className="flex-1 flex flex-col items-center justify-center py-8 text-center max-w-2xl mx-auto">
                    <div className="w-12 h-12 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400 mb-3 shadow-lg shadow-sky-500/10">
                      <Sparkles className="w-6 h-6" />
                    </div>
                    <h3 className="text-lg font-bold text-white mb-1">
                      Knowledge Base Ready
                    </h3>
                    <p className="text-xs text-slate-400 mb-6 max-w-md leading-relaxed">
                      Select a suggested inquiry below or ask any question about{" "}
                      <strong>{activeDoc.title || activeDoc.url}</strong>.
                    </p>

                    {/* Context-Aware Suggested Question Cards */}
                    <div className="grid grid-cols-1 gap-2.5 w-full text-left">
                      {getSuggestedQuestions(activeDoc, messages).map(
                        (chip, qIdx) => (
                          <button
                            key={qIdx}
                            onClick={() =>
                              handleSendMessage(
                                chip.replace(/^[📌🔍⚡📊]\s*/, "").trim(),
                              )
                            }
                            disabled={isGenerating}
                            className="p-3.5 rounded-xl bg-slate-900/80 hover:bg-sky-950/60 border border-white/10 hover:border-sky-500/40 text-xs text-slate-200 transition-all flex items-center justify-between group shadow-md hover:shadow-sky-500/10">
                            <span className="font-medium">{chip}</span>
                            <ArrowRight className="w-4 h-4 text-sky-400 opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
                          </button>
                        ),
                      )}
                    </div>
                  </div>
                )}

                {/* Messages Turn List */}
                {messages.map((msg, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex gap-3.5 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    {msg.role === "assistant" && (
                      <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shrink-0 text-white shadow-md shadow-sky-500/20 mt-1">
                        <Sparkles className="w-4 h-4" />
                      </div>
                    )}

                    <div
                      className={`max-w-[85%] rounded-2xl p-4 md:p-5 text-sm leading-relaxed ${
                        msg.role === "user"
                          ? "bg-slate-800/80 border border-slate-700/50 text-slate-100 shadow-xl backdrop-blur-md rounded-tr-[4px]"
                          : "bg-slate-900/60 backdrop-blur-sm border border-white/10 text-slate-200 shadow-xl rounded-tl-[4px]"
                      }`}>
                      {/* Agentic Chain-of-Thought / Thinking Accordion */}
                      {msg.role === "assistant" &&
                        msg.steps &&
                        msg.steps.length > 0 && (
                          <div className="mb-3 rounded-lg border border-white/10 bg-black/30 overflow-hidden text-xs">
                            <button
                              onClick={() =>
                                setExpandedThinking((prev) => ({
                                  ...prev,
                                  [idx]: !prev[idx],
                                }))
                              }
                              className="w-full flex items-center justify-between p-2 text-slate-400 hover:text-slate-200 bg-white/[0.02] transition-colors">
                              <span className="flex items-center gap-1.5 font-medium">
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                Agentic Reasoning Trace ({msg.steps.length}{" "}
                                steps)
                              </span>
                              {expandedThinking[idx] ? (
                                <ChevronDown className="w-3.5 h-3.5" />
                              ) : (
                                <ChevronRight className="w-3.5 h-3.5" />
                              )}
                            </button>
                            {expandedThinking[idx] && (
                              <div className="p-2.5 border-t border-white/5 space-y-1.5 font-sans text-[11px]">
                                {msg.steps.map((step, sIdx) => {
                                  const title =
                                    step.title ||
                                    step.action ||
                                    step.name ||
                                    (step.step
                                      ? step.step.toUpperCase()
                                      : `Step #${sIdx + 1}`);
                                  const detail =
                                    step.detail ||
                                    step.input_summary ||
                                    step.description ||
                                    "";
                                  return (
                                    <div
                                      key={sIdx}
                                      className="flex items-start gap-2 text-slate-300">
                                      <span className="text-sky-400 shrink-0 font-mono font-bold">
                                        #{sIdx + 1}
                                      </span>
                                      <div>
                                        <span className="text-slate-200 font-semibold">
                                          {title}
                                        </span>
                                        {detail && (
                                          <span className="text-slate-400 font-normal">
                                            {" — "}
                                            {detail}
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        )}

                      {/* Message Body Content */}
                      {msg.isThinking && !msg.content ? (
                        <div className="flex items-center gap-2 text-sky-400 py-1">
                          <RefreshCcw className="w-3.5 h-3.5 animate-spin" />
                          <span className="text-xs">
                            Reasoning and synthesising grounded answer...
                          </span>
                        </div>
                      ) : msg.role === "user" ? (
                        <div className="text-slate-100 whitespace-pre-wrap break-words overflow-hidden font-normal leading-relaxed">
                          {msg.content}
                        </div>
                      ) : (
                        <div className="chat-markdown">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              table({ children }) {
                                return (
                                  <div className="overflow-x-auto my-3 rounded-xl border border-white/10 shadow-lg">
                                    <table className="w-full text-xs text-left text-slate-200 border-collapse">
                                      {children}
                                    </table>
                                  </div>
                                );
                              },
                              thead({ children }) {
                                return (
                                  <thead className="bg-sky-950/60 text-sky-400 font-semibold uppercase text-[11px] border-b border-white/10">
                                    {children}
                                  </thead>
                                );
                              },
                              tbody({ children }) {
                                return (
                                  <tbody className="divide-y divide-white/5 bg-black/40">
                                    {children}
                                  </tbody>
                                );
                              },
                              tr({ children }) {
                                return (
                                  <tr className="hover:bg-white/[0.03] transition-colors">
                                    {children}
                                  </tr>
                                );
                              },
                              th({ children }) {
                                return (
                                  <th className="px-3.5 py-2.5 font-semibold text-sky-300">
                                    {children}
                                  </th>
                                );
                              },
                              td({ children }) {
                                return (
                                  <td className="px-3.5 py-2 leading-relaxed text-slate-300">
                                    {children}
                                  </td>
                                );
                              },
                              img({ node, src, alt, ...props }) {
                                if (!src) return null;
                                return (
                                  <div className="my-4 rounded-xl overflow-hidden border border-white/10 shadow-2xl bg-black/50 max-w-[540px]">
                                    <a
                                      href={src}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="block group relative cursor-zoom-in"
                                      title="Click to view full size image">
                                      <img
                                        src={src}
                                        alt={alt || "Article Illustration"}
                                        className="w-full max-h-[440px] object-contain rounded-t-xl bg-[#090b10] transition-transform duration-200 group-hover:scale-[1.01]"
                                        loading="lazy"
                                        referrerPolicy="no-referrer"
                                        {...props}
                                      />
                                    </a>
                                    {alt && alt !== "Content Diagram/Image" && (
                                      <div className="p-2.5 flex items-center justify-between text-[11px] text-slate-400 border-t border-white/5 bg-white/[0.02]">
                                        <span className="font-medium truncate mr-2">
                                          📊 {alt}
                                        </span>
                                        <a
                                          href={src}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="text-sky-400 hover:text-sky-300 shrink-0 text-[10px] underline">
                                          Open full size ↗
                                        </a>
                                      </div>
                                    )}
                                  </div>
                                );
                              },
                              code({
                                node,
                                inline,
                                className,
                                children,
                                ...props
                              }) {
                                const match = /language-(\w+)/.exec(
                                  className || "",
                                );
                                const codeString = String(children).replace(
                                  /\n$/,
                                  "",
                                );
                                return !inline ? (
                                  <div className="relative group my-2 rounded-lg overflow-hidden border border-white/10 bg-[#08090d]">
                                    <div className="flex items-center justify-between px-3 py-1.5 bg-white/5 border-b border-white/10 text-[11px] text-slate-400">
                                      <span>{match ? match[1] : "Code"}</span>
                                      <button
                                        onClick={() =>
                                          copyCode(codeString, idx)
                                        }
                                        className="flex items-center gap-1 text-slate-400 hover:text-white">
                                        {copiedCodeIdx === idx ? (
                                          <Check className="w-3 h-3 text-emerald-400" />
                                        ) : (
                                          <Copy className="w-3 h-3" />
                                        )}
                                        <span>
                                          {copiedCodeIdx === idx
                                            ? "Copied"
                                            : "Copy"}
                                        </span>
                                      </button>
                                    </div>
                                    <pre className="p-3 text-xs overflow-x-auto text-cyan-200">
                                      <code>{children}</code>
                                    </pre>
                                  </div>
                                ) : (
                                  <code className={className} {...props}>
                                    {children}
                                  </code>
                                );
                              },
                            }}>
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      )}

                      {/* Assistant Action Buttons */}
                      {msg.role === "assistant" && msg.content && (
                        <div className="flex items-center gap-3 mt-3 pt-2 border-t border-white/5 text-slate-500 text-xs">
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(msg.content);
                              alert("Copied to clipboard!");
                            }}
                            className="hover:text-slate-300 flex items-center gap-1 transition-colors">
                            <Copy className="w-3 h-3" /> Copy
                          </button>
                          <button
                            onClick={() =>
                              setFeedback((prev) => ({ ...prev, [idx]: "up" }))
                            }
                            className={`hover:text-emerald-400 flex items-center gap-1 transition-colors ${feedback[idx] === "up" ? "text-emerald-400" : ""}`}>
                            <ThumbsUp className="w-3 h-3" /> Helpful
                          </button>
                          <button
                            onClick={() =>
                              setFeedback((prev) => ({
                                ...prev,
                                [idx]: "down",
                              }))
                            }
                            className={`hover:text-rose-400 flex items-center gap-1 transition-colors ${feedback[idx] === "down" ? "text-rose-400" : ""}`}>
                            <ThumbsDown className="w-3 h-3" /> Unhelpful
                          </button>
                        </div>
                      )}
                    </div>

                    {msg.role === "user" && (
                      <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-600 to-cyan-500 border border-sky-300/40 flex items-center justify-center shrink-0 text-white font-bold text-xs shadow-md shadow-sky-500/25 mt-1">
                        U
                      </div>
                    )}
                  </motion.div>
                ))}

                <div className="h-10 shrink-0" ref={messagesEndRef} />
              </div>
            </div>
          )}
        </main>

        {/* Floating Chat Input Dock */}
        {(activeDoc || messages.length > 0) && (
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-[#08090c] via-[#08090c]/95 to-transparent backdrop-blur-md pb-5 shrink-0 z-20">
            <div className="max-w-4xl mx-auto">
              {/* Dynamic Context-Aware Suggested Prompt Chips (Only shown during active conversation to avoid duplication) */}
              {messages.length > 0 && (
                <div className="flex flex-wrap items-center gap-2 mb-2.5">
                  {getSuggestedQuestions(activeDoc, messages).map((chip) => (
                    <button
                      key={chip}
                      onClick={() =>
                        handleSendMessage(
                          chip.replace(/^[📌🔍⚡📊]\s*/, "").trim(),
                        )
                      }
                      disabled={isGenerating}
                      className="inline-flex items-center text-left leading-snug h-auto text-[11px] font-medium bg-white/5 hover:bg-sky-500/10 border border-white/10 hover:border-sky-500/30 text-slate-300 hover:text-sky-300 px-3.5 py-2 rounded-2xl transition-all duration-200 disabled:opacity-40 shadow-sm backdrop-blur-md">
                      {chip}
                    </button>
                  ))}
                </div>
              )}

              {/* Input Wrapper */}
              <div className="relative flex items-end gap-2 bg-slate-900/50 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl p-2 focus-within:bg-slate-900/80 focus-within:border-sky-500/50 focus-within:shadow-[0_0_30px_rgba(14,165,233,0.15)] transition-all duration-300">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder="Ask anything about the extracted web content..."
                  className="flex-1 max-h-36 min-h-[48px] bg-transparent resize-none outline-none p-3 text-sm text-slate-200 placeholder:text-slate-500 leading-relaxed font-sans"
                  rows={1}
                />

                {/* Send Button */}
                <button
                  onClick={() => handleSendMessage()}
                  disabled={!inputMessage.trim() || isGenerating}
                  className="w-11 h-11 shrink-0 rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white flex items-center justify-center transition-all disabled:opacity-40 disabled:cursor-not-allowed mb-0.5 mr-0.5 shadow-lg shadow-sky-500/25">
                  {isGenerating ? (
                    <RefreshCcw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4 ml-0.5" />
                  )}
                </button>
              </div>

              <div className="text-center mt-2 text-[10px] text-slate-500 font-medium">
                Press{" "}
                <kbd className="px-1 py-0.5 bg-white/5 border border-white/10 rounded">
                  Enter
                </kbd>{" "}
                to send •{" "}
                <kbd className="px-1 py-0.5 bg-white/5 border border-white/10 rounded">
                  Shift + Enter
                </kbd>{" "}
                for new line
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Document Inspector Modal */}
      <AnimatePresence>
        {showDocModal && activeDoc && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="glass-card border border-white/15 rounded-2xl p-6 max-w-2xl w-full shadow-2xl max-h-[85vh] flex flex-col">
              <div className="flex items-center justify-between pb-3 mb-3 border-b border-white/10">
                <div className="flex items-center gap-2 font-bold text-white text-sm">
                  <FileText className="w-4 h-4 text-sky-400" /> Document Content
                  Inspector
                </div>
                <button
                  onClick={() => setShowDocModal(false)}
                  className="text-slate-400 hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-3 flex-1 overflow-y-auto pr-1">
                <div className="p-3 bg-black/30 rounded-lg border border-white/5 space-y-1.5 text-xs">
                  <div>
                    <strong>Title:</strong> {activeDoc.title}
                  </div>
                  <div className="truncate">
                    <strong>URL:</strong>{" "}
                    <a
                      href={activeDoc.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sky-400 underline">
                      {activeDoc.url}
                    </a>
                  </div>
                  <div>
                    <strong>Word Count:</strong>{" "}
                    {activeDoc.wordCount?.toLocaleString()} words
                  </div>
                  <div>
                    <strong>Strategy Used:</strong>{" "}
                    <span className="uppercase text-emerald-400 font-semibold">
                      {activeDoc.strategyUsed}
                    </span>
                  </div>
                  <div>
                    <strong>Paywall Status:</strong>{" "}
                    {activeDoc.paywallBypassed
                      ? "Bypassed via Apollo/Jina"
                      : "Clean Web Extraction"}
                  </div>
                </div>
                <div>
                  <span className="text-xs font-medium text-slate-400 block mb-1">
                    Extracted Text Content Preview:
                  </span>
                  <div className="p-3 bg-black/40 rounded-lg border border-white/5 text-xs text-slate-300 font-mono max-h-72 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                    {activeDoc.content ||
                      "No raw text available for this session."}
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Model Catalog Modal */}
      <AnimatePresence>
        {showModelModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="glass-card border border-white/15 rounded-2xl p-6 max-w-xl w-full shadow-2xl max-h-[85vh] flex flex-col">
              <div className="flex items-center justify-between pb-3 mb-3 border-b border-white/10">
                <div className="flex items-center gap-2 font-bold text-white text-sm">
                  <Cpu className="w-4 h-4 text-sky-400" /> 10-Tier Resilient LLM
                  Cascade
                </div>
                <button
                  onClick={() => setShowModelModal(false)}
                  className="text-slate-400 hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>
              <p className="text-xs text-slate-300 mb-3">
                Queries dynamically route through a prioritized fallback chain
                across Google Gemini and Groq with sliding-window RPM rate
                limiters.
              </p>
              <div className="space-y-2 flex-1 overflow-y-auto pr-1">
                {(models.length > 0
                  ? models
                  : [
                      {
                        priority: 1,
                        model_name: "gemini-3.6-flash",
                        provider: "google",
                        status: "Healthy (Primary)",
                      },
                      {
                        priority: 2,
                        model_name: "openai/gpt-oss-120b",
                        provider: "groq",
                        status: "Active Fallback",
                      },
                      {
                        priority: 3,
                        model_name: "gemini-3.5-flash",
                        provider: "google",
                        status: "Standby",
                      },
                      {
                        priority: 4,
                        model_name: "groq/compound",
                        provider: "groq",
                        status: "Standby",
                      },
                      {
                        priority: 5,
                        model_name: "gemini-3.5-flash-lite",
                        provider: "google",
                        status: "Standby",
                      },
                      {
                        priority: 6,
                        model_name: "openai/gpt-oss-20b",
                        provider: "groq",
                        status: "Standby",
                      },
                    ]
                ).map((m, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-black/30 border border-white/5 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-sky-500/15 text-sky-300 flex items-center justify-center font-bold text-[10px]">
                        {m.priority || idx + 1}
                      </span>
                      <div>
                        <div className="font-semibold text-white">
                          {m.model_name || m.name}
                        </div>
                        <div className="text-[10px] text-slate-500 uppercase">
                          {m.provider}
                        </div>
                      </div>
                    </div>
                    <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-medium">
                      {m.status || "Operational"}
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
