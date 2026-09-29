import { useState, useEffect, useRef } from "react";
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
  Square,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = (() => {
  const envUrl =
    import.meta.env.VITE_API_BASE_URL ||
    import.meta.env.VITE_API_URL ||
    import.meta.env.VITE_API_BASE;

  if (!envUrl) return "http://localhost:8000/api";

  const trimmed = envUrl.trim().replace(/\/+$/, "");
  if (!trimmed.endsWith("/api") && !trimmed.includes("/api/")) {
    return `${trimmed}/api`;
  }
  return trimmed;
})();

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

function cleanMarkdownContent(raw) {
  if (!raw || typeof raw !== "string") return "";
  let text = raw;

  // 1. Remove malformed/doubled HTML break tags like <<brbr>>, <br>, &lt;br&gt;, etc.
  text = text.replace(/(&lt;|<)\s*<?\s*b\s*r\s*b?\s*r?\s*\/?>*(&gt;|>)+/gi, "\n");
  text = text.replace(/<\s*br\s*\/?>/gi, "\n");
  text = text.replace(/&lt;\s*br\s*\/?&gt;/gi, "\n");

  // 2. Fix duplicated pipe characters in Markdown tables (e.g. "||" -> "|")
  text = text.replace(/\|{2,}/g, "|");

  return text;
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

function normalizeText(text) {
  if (!text || typeof text !== "string") return "";
  return text
    .toLowerCase()
    .replace(/[?!.:;,`'"$#*()[\]{}]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Extracts real section headings and main topics from the website/document content.
 */
function extractDocumentTopics(docContent) {
  if (!docContent || typeof docContent !== "string") return [];
  const lines = docContent.split("\n");
  const topics = [];
  const seen = new Set();

  for (const rawLine of lines) {
    if (topics.length >= 15) break;
    const line = rawLine.trim();
    const headerMatch = line.match(/^#{1,4}\s+(.+)$/);
    if (headerMatch) {
      let title = headerMatch[1]
        .replace(/[*_#]/g, "")
        .replace(/[:\-–—]+$/, "")
        .trim();
      const norm = normalizeText(title);
      if (
        title.length >= 4 &&
        title.length <= 65 &&
        !seen.has(norm) &&
        !/^(table of contents|contents|introduction|overview|conclusion|summary|references|sources|navigation|footer|menu|share this|related articles|sign in|log in|subscribe)/i.test(
          norm,
        )
      ) {
        seen.add(norm);
        topics.push(title);
      }
    }
  }

  // Fallback to bold terms if few headers found
  if (topics.length < 3) {
    const boldMatches = docContent.match(/\*\*([A-Za-z0-9\s\-–—/]{4,40})\*\*/g) || [];
    for (const b of boldMatches) {
      if (topics.length >= 8) break;
      const clean = b.replace(/\*\*/g, "").replace(/[$`]/g, "").trim();
      const norm = normalizeText(clean);
      if (
        clean.length >= 4 &&
        clean.length <= 35 &&
        !seen.has(norm) &&
        !/^(note|tip|warning|important|caution|step|example|figure|summary|overview)/i.test(
          clean,
        )
      ) {
        seen.add(norm);
        topics.push(clean);
      }
    }
  }

  return topics;
}

/**
 * Extracts new key concepts, technologies, and terms introduced in the assistant response.
 */
function extractAnswerConcepts(content) {
  if (!content || typeof content !== "string") return [];
  const concepts = [];
  const seen = new Set();
  const boldMatches = content.match(/\*\*([A-Za-z0-9\s\-–—/]{3,40})\*\*/g) || [];
  for (const b of boldMatches) {
    const clean = b.replace(/\*\*/g, "").replace(/[$`]/g, "").trim();
    const norm = normalizeText(clean);
    if (
      clean.length >= 3 &&
      clean.length <= 35 &&
      !seen.has(norm) &&
      !/^(note|tip|warning|important|caution|step|example|figure|summary|overview|with constant)/i.test(
        clean,
      )
    ) {
      seen.add(norm);
      concepts.push(clean);
    }
  }
  return concepts;
}

/**
 * Extracts the core subject from user query by stripping common interrogative phrasing.
 */
function extractQuerySubject(query) {
  if (!query || typeof query !== "string") return "";
  return query
    .replace(
      /^(what is|what are|how does|how do|how can|why is|why are|tell me about|explain|can you explain|summarize|give me an overview of|compare|difference between)\s+/i,
      "",
    )
    .replace(
      /\s+(work under the hood|operate in this system|operate|work in practice|work|mean|differ|compare)\s*\??$/i,
      "",
    )
    .replace(/[?!.]/g, "")
    .trim();
}

/**
 * Generates dynamically evolving, context-aware suggested follow-up questions
 * directly based on:
 * 1. The website content (sections, headings, topics from activeDoc)
 * 2. The user's query and conversation history (avoiding repeat topics)
 * 3. The assistant's response (deep diving into newly introduced concepts)
 */
function getSuggestedQuestions(activeDoc, messages = []) {
  const docContent = activeDoc?.content || "";
  const rawTitle = (activeDoc?.title || activeDoc?.url || "").trim();
  const cleanTitle =
    rawTitle
      .replace(/^Domain Crawl \(\d+ pages\)/i, "")
      .replace(/^Site Crawl:/i, "")
      .trim() || "this document";
  const shortTitle =
    cleanTitle.length > 32 ? cleanTitle.substring(0, 30) + "..." : cleanTitle;
  const docTopics = extractDocumentTopics(docContent);

  // Active chat follow-ups
  if (messages && messages.length > 0) {
    const userMsgs = messages.filter((m) => m.role === "user");
    const assistantMsgs = messages.filter((m) => m.role === "assistant");
    const lastUserQuery =
      userMsgs.length > 0 ? userMsgs[userMsgs.length - 1].content || "" : "";
    const lastAssistant =
      assistantMsgs.length > 0 ? assistantMsgs[assistantMsgs.length - 1] : null;
    const lastContent = lastAssistant?.content || "";

    // If currently thinking without content, do not show suggestions
    if (lastAssistant?.isThinking && !lastContent) {
      return [];
    }

    const askedQueries = userMsgs.map((m) => normalizeText(m.content));
    const isTopicAsked = (topic) => {
      const normTopic = normalizeText(topic);
      if (!normTopic) return false;
      return askedQueries.some(
        (q) => q.includes(normTopic) || normTopic.includes(q),
      );
    };

    const querySubject = extractQuerySubject(lastUserQuery);
    const answerConcepts = extractAnswerConcepts(lastContent);
    const normLastQuery = normalizeText(lastUserQuery);

    const newConcepts = answerConcepts.filter(
      (c) => !normLastQuery.includes(normalizeText(c)),
    );

    const unaskedDocTopics = docTopics.filter(
      (t) => !isTopicAsked(t) && !normLastQuery.includes(normalizeText(t)),
    );

    const suggestions = [];
    const usedSubjects = new Set();
    const turnIndex = Math.max(0, userMsgs.length - 1);

    // 1. Concept deep dive from the answer
    let q1 = "";
    if (newConcepts.length > 0) {
      const topConcept = newConcepts[0];
      const q1Templates = [
        `🔍 How does "${topConcept}" operate in this architecture?`,
        `🔍 Can you explain the role and mechanics of "${topConcept}"?`,
        `🔍 What are the advantages of using "${topConcept}" here?`,
      ];
      q1 = q1Templates[turnIndex % q1Templates.length];
      usedSubjects.add(normalizeText(topConcept));
    } else if (querySubject) {
      q1 = `🔍 How does ${querySubject} work in practice under the hood?`;
      usedSubjects.add(normalizeText(querySubject));
    } else {
      q1 = `🔍 What are the underlying technical mechanisms?`;
    }
    suggestions.push(q1);

    // 2. Practical trade-offs, benchmarks, or implementation
    const combined = (lastUserQuery + " " + lastContent).toLowerCase();
    let q2 = "";
    const codeAngles = [
      "⚡ What are the key configuration options and common pitfalls?",
      "⚡ How can this implementation be optimized or scaled further?",
      "⚡ What edge cases and error handling steps should be included?",
      "⚡ Show an example testing or verifying this code snippet",
    ];
    const tradeOffAngles = [
      "⚡ What are the performance and latency trade-offs reported?",
      "⚡ How does this approach compare with alternatives in practical tests?",
      "⚡ What are the main edge cases or limitations mentioned in the article?",
      "⚡ What real-world production challenges are discussed?",
    ];

    if (
      combined.includes("code") ||
      combined.includes("python") ||
      combined.includes("api")
    ) {
      q2 = codeAngles[turnIndex % codeAngles.length];
    } else if (
      combined.includes("benchmark") ||
      combined.includes("latency") ||
      combined.includes("score") ||
      combined.includes("mrr")
    ) {
      q2 = "⚡ What are the performance and latency trade-offs reported?";
    } else if (
      newConcepts.length > 1 &&
      !usedSubjects.has(normalizeText(newConcepts[1]))
    ) {
      q2 = `⚡ What are the trade-offs and limitations of "${newConcepts[1]}"?`;
      usedSubjects.add(normalizeText(newConcepts[1]));
    } else if (querySubject && !usedSubjects.has(normalizeText(querySubject))) {
      q2 = `⚡ What are the practical limitations or trade-offs of ${querySubject}?`;
      usedSubjects.add(normalizeText(querySubject));
    } else {
      q2 = tradeOffAngles[turnIndex % tradeOffAngles.length];
    }
    suggestions.push(q2);

    // 3. Next unasked topic from the website content (rotates through unexplored sections)
    let q3 = "";
    const freshDocTopics = unaskedDocTopics.filter(
      (t) => !usedSubjects.has(normalizeText(t)),
    );
    const availableDocTopic =
      freshDocTopics.length > 0
        ? freshDocTopics[turnIndex % freshDocTopics.length]
        : null;

    if (availableDocTopic) {
      const q3Templates = [
        `📌 What does the document explain about "${availableDocTopic}"?`,
        `📌 How does "${availableDocTopic}" relate to this discussion?`,
        `📌 Tell me more about the section on "${availableDocTopic}"`,
      ];
      q3 = q3Templates[turnIndex % q3Templates.length];
      usedSubjects.add(normalizeText(availableDocTopic));
    } else if (
      newConcepts.length > 2 &&
      !usedSubjects.has(normalizeText(newConcepts[2]))
    ) {
      q3 = `📌 Can you elaborate on "${newConcepts[2]}"?`;
    } else {
      q3 = `📌 What are the primary conclusions or next steps from ${shortTitle}?`;
    }
    suggestions.push(q3);

    return suggestions;
  }

  // Pre-chat empty state (when document is loaded but no messages sent yet)
  if (docTopics.length >= 2) {
    const list = [
      `📌 Overview: Summarize the core thesis and key findings of "${shortTitle}"`,
      `🔍 Deep Dive: What does the article explain about "${docTopics[0]}"?`,
      `⚡ Technical Analysis: How does "${docTopics[1]}" operate according to the document?`,
    ];
    if (docTopics.length >= 3) {
      list.push(
        `📊 Trade-offs: What are the key findings or benchmarks for "${docTopics[2]}"?`,
      );
    }
    return list;
  }

  if (!activeDoc) {
    return [
      "📌 Summarize core takeaways",
      "🔍 Extract key technical architecture",
      "📊 What are the limitations or trade-offs?",
    ];
  }

  return [
    `📌 What are the primary takeaways and key findings from "${shortTitle}"?`,
    `🔍 How does the core methodology or architecture described here work?`,
    `⚡ What are the practical implications and trade-offs highlighted in this document?`,
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
  const [currentSessionId, setCurrentSessionId] = useState(() => {
    return localStorage.getItem("webchat_active_session_id") || null;
  });
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

  // Chat State (instant rehydration from localStorage to prevent loss on page refresh)
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem("webchat_active_messages");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [inputMessage, setInputMessage] = useState("");
  const [isStreaming, setIsStreaming] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [expandedThinking, setExpandedThinking] = useState({});
  const [copiedCodeIdx, setCopiedCodeIdx] = useState(null);
  const [copiedMsgIdx, setCopiedMsgIdx] = useState(null);
  const [feedback, setFeedback] = useState({});
  const abortControllerRef = useRef(null);

  // Keep active messages in sync with localStorage
  useEffect(() => {
    try {
      if (messages && messages.length > 0) {
        const cleanToStore = messages
          .filter((m) => m.content || (m.steps && m.steps.length > 0))
          .map((m) => ({
            role: m.role,
            content: m.content,
            citations: m.citations || [],
            modelInfo: m.modelInfo || "",
            provider: m.provider || "",
            steps: m.steps || [],
          }));
        if (cleanToStore.length > 0) {
          localStorage.setItem("webchat_active_messages", JSON.stringify(cleanToStore));
        }
      } else {
        localStorage.removeItem("webchat_active_messages");
      }
    } catch (e) {
      console.debug("Could not cache messages in localStorage", e);
    }
  }, [messages]);

  // Keep active session_id in sync with localStorage
  useEffect(() => {
    if (currentSessionId) {
      localStorage.setItem("webchat_active_session_id", currentSessionId);
    } else {
      localStorage.removeItem("webchat_active_session_id");
    }
  }, [currentSessionId]);

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

      // Transition to Chat View cleanly
      setMessages([]);
    } catch (err) {
      alert("Extraction Error: " + err.message);
    } finally {
      setIsIngesting(false);
      setIngestStatus("");
    }
  };

  // Stop Generating Handler
  const handleStopGenerating = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    setMessages((prev) => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (last && last.role === "assistant") {
        last.isThinking = false;
        if (!last.content) {
          last.content = "_Generation stopped by user._";
        }
      }
      return updated;
    });
  };

  // Copy Message Handler (with inline feedback)
  const handleCopyMessage = (content, idx) => {
    if (!content) return;
    navigator.clipboard.writeText(content);
    setCopiedMsgIdx(idx);
    setTimeout(() => {
      setCopiedMsgIdx((prev) => (prev === idx ? null : prev));
    }, 2000);
  };

  // Send Message
  const handleSendMessage = async (customQuery = null) => {
    const query = (customQuery || inputMessage).trim();
    if (!query || isGenerating) return;

    if (quota && !quota.can_request) {
      alert("Daily query limit reached. Quota resets at midnight UTC.");
      return;
    }

    // Extract recent conversation history for multi-turn context
    const historyPayload = messages
      .filter(
        (m) =>
          m.content &&
          !m.isThinking &&
          (m.role === "user" || m.role === "assistant"),
      )
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));

    if (!customQuery) setInputMessage("");
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setIsGenerating(true);

    // Placeholder for assistant turn
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

    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    try {
      const payload = {
        query: query,
        url: activeDoc?.url || validUrls.join(", ") || null,
        document_content: activeDoc?.content || null,
        chat_history: historyPayload,
        stream: isStreaming,
        session_id: currentSessionId,
        client_id: clientId,
      };

      if (isStreaming) {
        const response = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal,
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

                if (currentEvent === "step") {
                  setMessages((prev) => {
                    if (prev.length === 0) return prev;
                    const last = prev[prev.length - 1];
                    if (!last || last.role !== "assistant") return prev;
                    return [
                      ...prev.slice(0, -1),
                      {
                        ...last,
                        steps: [...(last.steps || []), parsed],
                      },
                    ];
                  });
                } else if (currentEvent === "citations") {
                  setMessages((prev) => {
                    if (prev.length === 0) return prev;
                    const last = prev[prev.length - 1];
                    if (!last || last.role !== "assistant") return prev;
                    return [
                      ...prev.slice(0, -1),
                      {
                        ...last,
                        citations: parsed,
                      },
                    ];
                  });
                } else if (parsed.chunk !== undefined) {
                  fullAnswer += parsed.chunk;
                  const currentText = fullAnswer;
                  setMessages((prev) => {
                    if (prev.length === 0) return prev;
                    const last = prev[prev.length - 1];
                    if (!last || last.role !== "assistant") return prev;
                    return [
                      ...prev.slice(0, -1),
                      {
                        ...last,
                        isThinking: false,
                        content: currentText,
                        modelInfo: parsed.model_used || last.modelInfo,
                        provider: parsed.provider || last.provider,
                        latency: parsed.latency_sec || last.latency,
                      },
                    ];
                  });
                }
              } catch (e) {
                // Ignore raw chunk parsing error
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
          signal,
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
      if (err.name === "AbortError") {
        return;
      }
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
      abortControllerRef.current = null;
      setIsGenerating(false);
      fetchQuota();
      fetchSessions();
    }
  };

  // Start Fresh New Chat (clears all active chat history and doc, returns to home screen)
  const startNewChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    localStorage.removeItem("webchat_active_session_id");
    localStorage.removeItem("webchat_active_messages");
    localStorage.removeItem("webchat_active_doc");
    setCurrentSessionId(null);
    setMessages([]);
    setActiveDoc(null);
    setUrlsInput("");
    setInputMessage("");
    setExpandedThinking({});
    setFeedback({});
    fetchSessions();
  };

  // Start New Chat with Current Document (clears conversation history on the active document)
  const startNewChatForCurrentDoc = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    localStorage.removeItem("webchat_active_session_id");
    localStorage.removeItem("webchat_active_messages");
    setCurrentSessionId(null);
    setMessages([]);
    setInputMessage("");
    setExpandedThinking({});
    setFeedback({});
    fetchSessions();
  };

  // Alias for backward compatibility
  const startFreshWithNewUrl = startNewChat;

  // Load Session
  const loadSession = async (sid) => {
    if (!sid) return;
    try {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      setIsGenerating(false);

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
        } else {
          setActiveDoc(null);
          localStorage.removeItem("webchat_active_doc");
        }
        if (Array.isArray(data.messages)) {
          const loadedMsgs = data.messages.map((m) => ({
            role: m.role,
            content: m.content,
            citations: m.citations || [],
            modelInfo: m.model_used || "Cached",
            provider: m.provider || "",
            steps: [],
          }));
          setMessages(loadedMsgs);
          try {
            localStorage.setItem("webchat_active_messages", JSON.stringify(loadedMsgs));
          } catch (e) {}
        }
      } else if (res.status === 404) {
        // Stale or deleted session: clean up local storage
        localStorage.removeItem("webchat_active_session_id");
        localStorage.removeItem("webchat_active_messages");
        setCurrentSessionId(null);
        setMessages([]);
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
    <div className="flex h-screen w-full bg-[#212121] text-[#ececec] overflow-hidden font-sans selection:bg-zinc-700 selection:text-white">
      {/* Sidebar - ChatGPT Style Dark Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="h-full border-r border-[#2e2e33] bg-[#171717] flex flex-col shrink-0 z-30 select-none">
            {/* Sidebar Brand Header */}
            <div className="p-3.5 border-b border-[#27272a] flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-b from-zinc-700 to-zinc-900 border border-zinc-700/80 flex items-center justify-center shadow-sm text-white">
                  <Sparkles className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h1 className="font-semibold text-sm text-white tracking-tight">
                    WebChat
                  </h1>
                </div>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1.5 hover:bg-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors md:hidden">
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* New Session Action Buttons */}
            <div className="p-3 space-y-2">
              <button
                onClick={startNewChat}
                className="w-full flex items-center justify-center gap-2 bg-white hover:bg-zinc-200 text-black py-2.5 px-3 rounded-xl font-semibold text-xs shadow-md transition-all active:scale-[0.99]"
                title="Start a fresh chat conversation">
                <Plus className="w-4 h-4 text-black" /> New Chat
              </button>
              {activeDoc && (
                <button
                  onClick={startNewChatForCurrentDoc}
                  className="w-full flex items-center justify-center gap-2 bg-[#212121] hover:bg-[#27272a] text-zinc-300 hover:text-white py-2 px-3 rounded-xl font-medium text-[11px] border border-zinc-800 transition-all active:scale-[0.99]"
                  title="Clear messages but keep current document loaded">
                  <RefreshCcw className="w-3 h-3 text-zinc-400" /> New Chat (Same Doc)
                </button>
              )}
            </div>

            {/* Session Search */}
            <div className="px-3 pb-2">
              <div className="flex items-center gap-2 bg-[#212121] border border-zinc-800 focus-within:border-zinc-600 px-2.5 py-1.5 rounded-lg text-xs text-zinc-400 transition-colors">
                <Search className="w-3.5 h-3.5 text-zinc-500" />
                <input
                  type="text"
                  placeholder="Search chats..."
                  value={sessionSearch}
                  onChange={(e) => setSessionSearch(e.target.value)}
                  className="bg-transparent border-none outline-none text-zinc-200 text-xs w-full placeholder:text-zinc-500"
                />
              </div>
            </div>

            {/* Sessions List */}
            <div className="flex-1 overflow-y-auto px-2 py-1 space-y-0.5">
              <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider px-2.5 py-1.5">
                Chat History ({filteredSessions.length})
              </div>
              {filteredSessions.length === 0 ? (
                <div className="text-xs text-zinc-500 text-center py-8 italic">
                  No conversation history
                </div>
              ) : (
                filteredSessions.map((s) => (
                  <div
                    key={s.session_id}
                    onClick={() => loadSession(s.session_id)}
                    className={`group flex items-center justify-between px-2.5 py-2 rounded-lg text-xs cursor-pointer transition-colors ${
                      currentSessionId === s.session_id
                        ? "bg-[#212121] text-white font-medium border border-zinc-700/60"
                        : "hover:bg-[#212121]/60 text-zinc-400 hover:text-zinc-200"
                    }`}>
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <MessageSquare className="w-3.5 h-3.5 shrink-0 text-zinc-400 group-hover:text-zinc-300" />
                      <span className="truncate">
                        {s.title || s.url || "Untitled Chat"}
                      </span>
                    </div>
                    <button
                      onClick={(e) => deleteSession(e, s.session_id)}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:text-rose-400 text-zinc-500 rounded transition-opacity"
                      title="Delete chat">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))
              )}
            </div>

            {/* Device Quota & Plan Footer */}
            <div className="p-3 border-t border-[#27272a] bg-[#141414]">
              <div className="p-3 rounded-xl border border-zinc-800/80 bg-[#1a1a1c]">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Zap className="w-3.5 h-3.5 text-zinc-300" />
                    <span className="text-[11px] text-zinc-400 font-medium">
                      Daily Queries
                    </span>
                  </div>
                  <span className="text-[11px] text-white font-semibold tabular-nums">
                    {remainingQuota} / {totalQuota}
                  </span>
                </div>
                <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-zinc-400 to-white transition-all duration-500 rounded-full"
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
      <div className="flex-1 flex flex-col h-full overflow-hidden relative z-10 bg-[#212121]">
        {/* Top Navbar */}
        <header className="h-14 border-b border-[#2e2e33] bg-[#212121]/90 backdrop-blur-md flex items-center justify-between px-4 shrink-0 gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-1.5 hover:bg-zinc-800 rounded-lg text-zinc-400 hover:text-white transition-colors shrink-0"
              title="Toggle Sidebar">
              <Menu className="w-4 h-4" />
            </button>

            {activeDoc && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#2a2a2d] border border-zinc-700/80 text-xs text-zinc-200 min-w-0 max-w-xs md:max-w-md">
                <Globe className="w-3.5 h-3.5 text-zinc-300 shrink-0" />
                <span className="truncate font-medium">
                  {activeDoc.title || activeDoc.url}
                </span>
              </div>
            )}
          </div>

          {/* Right Header Space / Actions */}
          <div className="flex items-center gap-2 shrink-0">
            {activeDoc && (
              <button
                onClick={() => setShowDocModal(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800/80 hover:bg-zinc-700 border border-zinc-700 text-zinc-300 hover:text-white text-xs font-medium transition-all shadow-sm active:scale-95"
                title="Inspect Extracted Document">
                <FileText className="w-3.5 h-3.5 text-zinc-400" />
                <span className="hidden sm:inline">Inspector</span>
              </button>
            )}
            {activeDoc && (
              <>
                <button
                  onClick={startNewChat}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-zinc-200 text-black text-xs font-semibold transition-all shadow-sm active:scale-95"
                  title="Start a fresh new chat conversation">
                  <Plus className="w-3.5 h-3.5 text-black" />
                  <span className="hidden sm:inline">New Chat</span>
                </button>
                <button
                  onClick={startNewChatForCurrentDoc}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-200 hover:text-white text-xs font-medium transition-all shadow-sm active:scale-95"
                  title="Reset conversation on current document">
                  <RefreshCcw className="w-3.5 h-3.5 text-zinc-400" />
                  <span className="hidden sm:inline">Reset Chat</span>
                </button>
              </>
            )}
          </div>
        </header>

        {/* Content Stream / Home Screen */}
        <main className="flex-1 relative flex flex-col overflow-hidden">
          {messages.length === 0 && !activeDoc ? (
            /* Home Start Screen */
            <div className="flex-1 flex flex-col items-center justify-start pt-8 md:pt-14 p-4 md:p-8 max-w-3xl mx-auto w-full">
              {/* Hero Banner */}
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center mb-6">
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-b from-zinc-700 to-zinc-900 border border-white/10 flex items-center justify-center text-white mx-auto mb-4 shadow-lg">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <h2 className="text-2xl md:text-3xl font-bold text-white tracking-tight mb-2">
                  What can I help you extract and explore?
                </h2>
                <p className="text-zinc-400 text-sm max-w-lg mx-auto">
                  Paste website URLs to extract articles, bypass paywalls, crawl documentation, and chat with grounded AI.
                </p>
              </motion.div>

              {/* Multi-URL Ingestion Card */}
              <motion.div
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className="w-full bg-[#18181b] rounded-2xl p-5 border border-zinc-800 shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between pb-3 mb-3 border-b border-zinc-800">
                  <div className="flex items-center gap-2 text-xs font-medium text-zinc-300">
                    <Globe className="w-4 h-4 text-zinc-300" /> Target Website URL(s)
                  </div>
                  <div className="flex items-center gap-2">
                    {urlsInput && (
                      <button
                        onClick={() => setUrlsInput("")}
                        className="text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors">
                        Clear
                      </button>
                    )}
                    {invalidUrls.length > 0 && (
                      <span className="text-[11px] px-2.5 py-0.5 rounded-full font-medium bg-rose-500/10 text-rose-300 border border-rose-500/30 flex items-center gap-1">
                        <AlertCircle className="w-3 h-3 text-rose-400" />{" "}
                        {invalidUrls.length} Invalid
                      </span>
                    )}
                    <span
                      className={`text-[11px] px-2.5 py-0.5 rounded-full font-medium flex items-center gap-1 ${
                        validUrls.length > 0
                          ? "bg-zinc-800 text-zinc-200 border border-zinc-700"
                          : "bg-zinc-800/50 text-zinc-500"
                      }`}>
                      {validUrls.length > 0 && (
                        <CheckCircle2 className="w-3 h-3 text-zinc-300" />
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
                  className={`w-full h-32 bg-[#121214] border rounded-xl p-3.5 text-sm text-zinc-100 placeholder:text-zinc-600 resize-none outline-none transition-colors font-mono ${
                    invalidUrls.length > 0
                      ? "border-rose-500/40 focus:border-rose-500/60"
                      : "border-zinc-800 focus:border-zinc-600"
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
                <div className="flex items-center justify-between pt-3 mt-3 border-t border-zinc-800">
                  <div className="text-[11px] text-zinc-500">
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
                    className="bg-white hover:bg-zinc-200 text-black font-semibold text-xs px-5 py-2.5 rounded-xl shadow-md flex items-center gap-2 transition-all active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed">
                    {isIngesting ? (
                      <>
                        <RefreshCcw className="w-4 h-4 animate-spin text-black" />{" "}
                        Ingesting & Indexing...
                      </>
                    ) : (
                      <>
                        <Zap className="w-4 h-4 text-black" /> Ingest &amp; Start Chat
                      </>
                    )}
                  </button>
                </div>
              </motion.div>

              {/* Feature Highlights Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full mt-6">
                {[
                  {
                    icon: <Layers className="w-4 h-4 text-zinc-300" />,
                    title: "Agentic CRAG",
                    desc: "Query rewriting, web fallback, and groundedness checks with LangGraph.",
                  },
                  {
                    icon: <ShieldCheck className="w-4 h-4 text-zinc-300" />,
                    title: "Paywall Bypass",
                    desc: "Apollo State extraction, Jina headless reading, and Wayback Machine fallback.",
                  },
                  {
                    icon: <Cpu className="w-4 h-4 text-zinc-300" />,
                    title: "10-Tier Failover",
                    desc: "Gemini 3.6 Flash auto-cascading to Groq LLaMA models with circuit breakers.",
                  },
                ].map((feat, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-[#18181b]/70 border border-zinc-800 hover:border-zinc-700 transition-colors">
                    <div className="flex items-center gap-2 font-semibold text-xs text-zinc-200 mb-1">
                      {feat.icon} {feat.title}
                    </div>
                    <p className="text-[11px] text-zinc-400 leading-relaxed">
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
              <div className="p-4 md:p-6 pb-44 md:pb-48 space-y-6 max-w-4xl mx-auto w-full">
                {/* Context Summary Header */}
                {activeDoc && (
                  <div className="p-3 rounded-xl bg-[#18181b] border border-zinc-800 flex items-center justify-between text-xs text-zinc-300 gap-3">
                    <div className="flex items-center gap-2 min-w-0">
                      <Globe className="w-4 h-4 text-zinc-300 shrink-0" />
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
                        className="px-2.5 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700 text-zinc-200 hover:text-white font-medium text-[11px] transition-colors flex items-center gap-1.5 border border-zinc-700/60"
                        title="Clear conversation on this document">
                        <RefreshCcw className="w-3 h-3 text-zinc-400" />
                        <span>Reset Chat</span>
                      </button>
                      <button
                        onClick={startNewChat}
                        className="px-2.5 py-1 rounded-md bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 text-[11px] transition-colors border border-zinc-800"
                        title="Clear document and start fresh">
                        Change URL
                      </button>
                    </div>
                  </div>
                )}

                {/* Centered Quick Start Cards when document is ready but no message sent yet */}
                {messages.length === 0 && activeDoc && (
                  <div className="flex-1 flex flex-col items-center justify-center py-8 text-center max-w-2xl mx-auto">
                    <div className="w-12 h-12 rounded-2xl bg-gradient-to-b from-zinc-700 to-zinc-900 border border-white/10 flex items-center justify-center text-white mb-3 shadow-md">
                      <Sparkles className="w-6 h-6" />
                    </div>
                    <h3 className="text-lg font-bold text-white mb-1">
                      Knowledge Base Ready
                    </h3>
                    <p className="text-xs text-zinc-400 mb-6 max-w-md leading-relaxed">
                      Select a suggested inquiry below or ask any question about{" "}
                      <strong className="text-zinc-200">{activeDoc.title || activeDoc.url}</strong>.
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
                            className="p-3.5 rounded-xl bg-[#18181b] hover:bg-[#222226] border border-zinc-800 hover:border-zinc-700 text-xs text-zinc-200 transition-all flex items-center justify-between group shadow-sm">
                            <span className="font-medium text-zinc-200">{chip}</span>
                            <ArrowRight className="w-4 h-4 text-zinc-400 group-hover:text-white group-hover:translate-x-0.5 transition-all" />
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
                      <div className="w-8 h-8 rounded-full bg-gradient-to-b from-zinc-700 to-zinc-900 border border-white/20 flex items-center justify-center shrink-0 text-white shadow-md mt-0.5">
                        <Sparkles className="w-4 h-4 text-white" />
                      </div>
                    )}

                    <div
                      className={`text-sm leading-relaxed ${
                        msg.role === "user"
                          ? "bg-[#2f2f2f] text-white border border-zinc-700/60 rounded-3xl rounded-tr-md px-5 py-3.5 shadow-sm max-w-[85%] md:max-w-[75%]"
                          : "max-w-[90%] md:max-w-[85%] rounded-2xl p-4 md:p-5 text-[14.5px] text-[#ececec] bg-[#18181b]/80 border border-zinc-800/80 shadow-md backdrop-blur-sm rounded-tl-md"
                      }`}>
                      {/* Agentic Chain-of-Thought / Thinking Accordion */}
                      {msg.role === "assistant" &&
                        msg.steps &&
                        msg.steps.length > 0 && (
                          <div className="mb-3 rounded-xl border border-zinc-800 bg-[#121214] overflow-hidden text-xs">
                            <button
                              onClick={() =>
                                setExpandedThinking((prev) => ({
                                  ...prev,
                                  [idx]: !prev[idx],
                                }))
                              }
                              className="w-full flex items-center justify-between p-2.5 text-zinc-400 hover:text-zinc-200 bg-zinc-900/40 transition-colors">
                              <span className="flex items-center gap-1.5 font-medium">
                                <CheckCircle2 className="w-3.5 h-3.5 text-zinc-300" />
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
                              <div className="p-3 border-t border-zinc-800/80 space-y-2 font-sans text-[11px] bg-zinc-950/40">
                                {msg.steps.map((step, sIdx) => {
                                  const isLatestActive =
                                    msg.isThinking &&
                                    sIdx === msg.steps.length - 1;
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
                                      className={`flex items-start gap-2.5 transition-colors ${
                                        isLatestActive
                                          ? "text-zinc-100"
                                          : "text-zinc-300"
                                      }`}>
                                      <span className="text-zinc-500 shrink-0 font-mono font-bold">
                                        #{sIdx + 1}
                                      </span>
                                      <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2">
                                          <span className="font-semibold">
                                            {title}
                                          </span>
                                          {isLatestActive && (
                                            <span className="relative flex h-1.5 w-1.5">
                                              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-zinc-400 opacity-75"></span>
                                              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-zinc-200"></span>
                                            </span>
                                          )}
                                        </div>
                                        {detail && (
                                          <div className="text-zinc-400 font-normal mt-0.5 break-words">
                                            {detail}
                                          </div>
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
                        <div className="flex items-center gap-3 text-zinc-300 py-2.5 px-3 rounded-xl bg-zinc-900/60 border border-zinc-800/80 text-xs font-medium">
                          <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-zinc-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-zinc-200"></span>
                          </span>
                          <span className="text-zinc-300">
                            {msg.steps && msg.steps.length > 0
                              ? `Agent active: ${msg.steps[msg.steps.length - 1].title || msg.steps[msg.steps.length - 1].step || "Synthesizing"}`
                              : "Thinking & synthesizing grounded response..."}
                          </span>
                        </div>
                      ) : msg.role === "user" ? (
                        <div className="text-zinc-100 text-[14.5px] leading-relaxed whitespace-pre-wrap break-words font-normal">
                          {msg.content}
                        </div>
                      ) : (
                        <div className="chat-markdown">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              table({ children }) {
                                return (
                                  <div className="overflow-x-auto my-3 rounded-lg border border-zinc-800 shadow-md">
                                    <table className="w-full text-xs text-left text-zinc-200 border-collapse">
                                      {children}
                                    </table>
                                  </div>
                                );
                              },
                              thead({ children }) {
                                return (
                                  <thead className="bg-[#1f1f23] text-zinc-200 font-semibold text-xs border-b border-zinc-800">
                                    {children}
                                  </thead>
                                );
                              },
                              tbody({ children }) {
                                return (
                                  <tbody className="divide-y divide-zinc-800 bg-[#121214]">
                                    {children}
                                  </tbody>
                                );
                              },
                              tr({ children }) {
                                return (
                                  <tr className="hover:bg-zinc-800/30 transition-colors">
                                    {children}
                                  </tr>
                                );
                              },
                              th({ children }) {
                                return (
                                  <th className="px-3.5 py-2.5 font-semibold text-white">
                                    {children}
                                  </th>
                                );
                              },
                              td({ children }) {
                                return (
                                  <td className="px-3.5 py-2 leading-relaxed text-zinc-300">
                                    {children}
                                  </td>
                                );
                              },
                              img({ node, src, alt, ...props }) {
                                if (!src) return null;
                                return (
                                  <div className="my-4 rounded-xl overflow-hidden border border-zinc-800 shadow-2xl bg-[#0d0d0e] max-w-[560px]">
                                    <a
                                      href={src}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="block group relative cursor-zoom-in"
                                      title="Click to view full size image">
                                      <img
                                        src={src}
                                        alt={alt || "Article Illustration"}
                                        className="w-full max-h-[440px] object-contain rounded-t-xl bg-[#0d0d0e] transition-transform duration-200 group-hover:scale-[1.01]"
                                        loading="lazy"
                                        referrerPolicy="no-referrer"
                                        {...props}
                                      />
                                    </a>
                                    {alt && alt !== "Content Diagram/Image" && (
                                      <div className="p-2.5 flex items-center justify-between text-[11px] text-zinc-400 border-t border-zinc-800 bg-zinc-900/40">
                                        <span className="font-medium truncate mr-2">
                                          📊 {alt}
                                        </span>
                                        <a
                                          href={src}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="text-zinc-300 hover:text-white shrink-0 text-[10px] underline">
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
                                  <div className="relative group my-3 rounded-xl overflow-hidden border border-zinc-800 bg-[#0d0d0e]">
                                    <div className="flex items-center justify-between px-3.5 py-2 bg-[#212124] border-b border-[#2e2e33] text-xs text-zinc-400 font-mono">
                                      <span>{match ? match[1] : "code"}</span>
                                      <button
                                        onClick={() =>
                                          copyCode(codeString, idx)
                                        }
                                        className="flex items-center gap-1.5 text-zinc-400 hover:text-white transition-colors">
                                        {copiedCodeIdx === idx ? (
                                          <Check className="w-3.5 h-3.5 text-white" />
                                        ) : (
                                          <Copy className="w-3.5 h-3.5" />
                                        )}
                                        <span>
                                          {copiedCodeIdx === idx
                                            ? "Copied"
                                            : "Copy code"}
                                        </span>
                                      </button>
                                    </div>
                                    <pre className="p-4 text-xs overflow-x-auto text-[#f4f4f5] font-mono leading-relaxed bg-[#0d0d0e]">
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
                            {cleanMarkdownContent(msg.content)}
                          </ReactMarkdown>
                          {isGenerating && idx === messages.length - 1 && (
                            <span
                              className="inline-block w-1.5 h-4 bg-zinc-300 ml-1 translate-y-0.5 animate-pulse rounded-xs"
                              aria-hidden="true"
                            />
                          )}
                        </div>
                      )}

                      {/* Grounded Citations & Sources (if available) */}
                      {msg.role === "assistant" &&
                        msg.citations &&
                        msg.citations.length > 0 && (
                          <div className="mt-3.5 pt-3 border-t border-zinc-800/80">
                            <div className="text-[11px] font-medium text-zinc-400 mb-2 flex items-center gap-1.5">
                              <BookOpen className="w-3.5 h-3.5 text-zinc-300" />
                              <span>Sources & Citations:</span>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {msg.citations.map((c, cIdx) => (
                                <a
                                  key={cIdx}
                                  href={c.url || "#"}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#222226] hover:bg-[#2b2b30] border border-zinc-700/60 text-xs text-zinc-300 hover:text-white transition-colors"
                                  title={c.snippet || c.title || c.url}>
                                  <ExternalLink className="w-3 h-3 text-zinc-400" />
                                  <span className="truncate max-w-[200px]">
                                    {c.title || c.url || `Source #${cIdx + 1}`}
                                  </span>
                                </a>
                              ))}
                            </div>
                          </div>
                        )}

                      {/* Suggested Follow-Ups for latest completed assistant message */}
                      {idx === messages.length - 1 &&
                        msg.role === "assistant" &&
                        msg.content &&
                        !isGenerating && (
                          <div className="mt-3.5 pt-3 border-t border-zinc-800/80">
                            <div className="text-[11px] font-semibold text-zinc-400 mb-2 flex items-center gap-1.5 uppercase tracking-wider">
                              <Sparkles className="w-3.5 h-3.5 text-zinc-300" />
                              <span>Suggested Follow-Ups:</span>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {getSuggestedQuestions(activeDoc, messages).map(
                                (chip, qIdx) => (
                                  <button
                                    key={qIdx}
                                    onClick={() =>
                                      handleSendMessage(
                                        chip.replace(/^[📌🔍⚡📊]\s*/, "").trim(),
                                      )
                                    }
                                    className="inline-flex items-center text-left text-xs font-medium bg-[#222226] hover:bg-[#2b2b30] border border-zinc-700/60 hover:border-zinc-500 text-zinc-300 hover:text-white px-3 py-1.5 rounded-xl transition-all shadow-xs cursor-pointer group">
                                    <span className="mr-1.5 opacity-80 group-hover:opacity-100">
                                      {chip.slice(0, 2)}
                                    </span>
                                    <span>
                                      {chip.replace(/^[📌🔍⚡📊]\s*/, "")}
                                    </span>
                                  </button>
                                ),
                              )}
                            </div>
                          </div>
                        )}

                      {/* Assistant Action Buttons */}
                      {msg.role === "assistant" && msg.content && (
                        <div className="flex items-center justify-between mt-3 pt-2.5 border-t border-zinc-800/80 text-zinc-500 text-xs">
                          <div className="flex items-center gap-3">
                            <button
                              onClick={() => handleCopyMessage(cleanMarkdownContent(msg.content), idx)}
                              className="hover:text-zinc-200 flex items-center gap-1.5 transition-colors"
                              title="Copy response">
                              {copiedMsgIdx === idx ? (
                                <>
                                  <Check className="w-3.5 h-3.5 text-zinc-200" />
                                  <span className="text-zinc-200 font-medium">Copied!</span>
                                </>
                              ) : (
                                <>
                                  <Copy className="w-3.5 h-3.5" />
                                  <span>Copy</span>
                                </>
                              )}
                            </button>
                            <button
                              onClick={() =>
                                setFeedback((prev) => ({ ...prev, [idx]: "up" }))
                              }
                              className={`hover:text-white flex items-center gap-1.5 transition-colors ${feedback[idx] === "up" ? "text-white font-semibold" : ""}`}>
                              <ThumbsUp className="w-3.5 h-3.5" /> Helpful
                            </button>
                            <button
                              onClick={() =>
                                setFeedback((prev) => ({
                                  ...prev,
                                  [idx]: "down",
                                }))
                              }
                              className={`hover:text-rose-400 flex items-center gap-1.5 transition-colors ${feedback[idx] === "down" ? "text-rose-400" : ""}`}>
                              <ThumbsDown className="w-3.5 h-3.5" /> Unhelpful
                            </button>
                          </div>
                          {(msg.modelInfo || msg.latency) && (
                            <div className="text-[11px] text-zinc-500 flex items-center gap-1.5 font-mono">
                              {msg.modelInfo && (
                                <span>{formatModelName(msg.modelInfo)}</span>
                              )}
                              {msg.latency ? (
                                <span>• {Number(msg.latency).toFixed(1)}s</span>
                              ) : null}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {msg.role === "user" && (
                      <div className="w-8 h-8 rounded-full bg-zinc-700 border border-zinc-600/70 flex items-center justify-center shrink-0 text-white font-medium text-xs mt-0.5">
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

        {/* Floating Chat Input Dock - ChatGPT Signature Floating Input Bar */}
        {(activeDoc || messages.length > 0) && (
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-[#212121] via-[#212121]/95 to-transparent pb-5 shrink-0 z-20">
            <div className="max-w-4xl mx-auto">
              {/* Dynamic Context-Aware Suggested Prompt Chips */}
              {messages.length > 0 && !isGenerating && (
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
                      className="inline-flex items-center text-left text-xs font-medium bg-[#2a2a2d] hover:bg-[#343438] border border-zinc-700/70 text-zinc-300 hover:text-white px-3.5 py-1.5 rounded-full transition-all disabled:opacity-40 shadow-sm cursor-pointer hover:border-zinc-500">
                      {chip}
                    </button>
                  ))}
                </div>
              )}

              {/* Input Wrapper */}
              <div className="relative flex items-end gap-2 bg-[#2f2f2f] border border-zinc-700/70 rounded-3xl p-2 shadow-2xl focus-within:border-zinc-500 focus-within:bg-[#333333] transition-all">
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
                  className="flex-1 max-h-36 min-h-[48px] bg-transparent resize-none outline-none p-3 text-sm text-zinc-100 placeholder:text-zinc-500 leading-relaxed font-sans"
                  rows={1}
                />

                {/* Send / Stop Button */}
                {isGenerating ? (
                  <button
                    type="button"
                    onClick={handleStopGenerating}
                    title="Stop generating"
                    className="w-9 h-9 shrink-0 rounded-full bg-white hover:bg-zinc-200 text-black flex items-center justify-center mb-1 mr-1 transition-all active:scale-95 shadow-md cursor-pointer">
                    <Square className="w-3.5 h-3.5 fill-black text-black" />
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => handleSendMessage()}
                    disabled={!inputMessage.trim()}
                    className={`w-9 h-9 shrink-0 rounded-full flex items-center justify-center mb-1 mr-1 transition-all ${
                      !inputMessage.trim()
                        ? "bg-zinc-700/40 text-zinc-500 cursor-not-allowed"
                        : "bg-white text-black hover:bg-zinc-200 active:scale-95 shadow-md cursor-pointer"
                    }`}>
                    <Send className="w-4 h-4 ml-0.5" />
                  </button>
                )}
              </div>

              <div className="text-center mt-2 text-[11px] text-zinc-500 font-normal">
                Press{" "}
                <kbd className="px-1.5 py-0.5 bg-zinc-800 border border-zinc-700 rounded text-zinc-400 text-[10px]">
                  Enter
                </kbd>{" "}
                to send •{" "}
                <kbd className="px-1.5 py-0.5 bg-zinc-800 border border-zinc-700 rounded text-zinc-400 text-[10px]">
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
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-[#18181b] border border-zinc-800 rounded-2xl p-6 max-w-2xl w-full shadow-2xl max-h-[85vh] flex flex-col text-zinc-200">
              <div className="flex items-center justify-between pb-3 mb-3 border-b border-zinc-800">
                <div className="flex items-center gap-2 font-bold text-white text-sm">
                  <FileText className="w-4 h-4 text-zinc-300" /> Document Content
                  Inspector
                </div>
                <button
                  onClick={() => setShowDocModal(false)}
                  className="text-zinc-400 hover:text-white p-1 rounded-lg hover:bg-zinc-800 transition-colors">
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-3 flex-1 overflow-y-auto pr-1">
                <div className="p-3 bg-[#121214] rounded-xl border border-zinc-800 space-y-1.5 text-xs">
                  <div>
                    <strong className="text-white">Title:</strong> {activeDoc.title}
                  </div>
                  <div className="truncate">
                    <strong className="text-white">URL:</strong>{" "}
                    <a
                      href={activeDoc.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-zinc-200 hover:text-white underline">
                      {activeDoc.url}
                    </a>
                  </div>
                  <div>
                    <strong className="text-white">Word Count:</strong>{" "}
                    {activeDoc.wordCount?.toLocaleString()} words
                  </div>
                  <div>
                    <strong className="text-white">Strategy Used:</strong>{" "}
                    <span className="uppercase text-white font-semibold">
                      {activeDoc.strategyUsed}
                    </span>
                  </div>
                  <div>
                    <strong className="text-white">Paywall Status:</strong>{" "}
                    {activeDoc.paywallBypassed
                      ? "Bypassed via Apollo/Jina"
                      : "Clean Web Extraction"}
                  </div>
                </div>
                <div>
                  <span className="text-xs font-medium text-zinc-400 block mb-1">
                    Extracted Text Content Preview:
                  </span>
                  <div className="p-3 bg-[#0d0d0e] rounded-xl border border-zinc-800 text-xs text-zinc-300 font-mono max-h-72 overflow-y-auto whitespace-pre-wrap leading-relaxed">
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
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-[#18181b] border border-zinc-800 rounded-2xl p-6 max-w-xl w-full shadow-2xl max-h-[85vh] flex flex-col text-zinc-200">
              <div className="flex items-center justify-between pb-3 mb-3 border-b border-zinc-800">
                <div className="flex items-center gap-2 font-bold text-white text-sm">
                  <Cpu className="w-4 h-4 text-zinc-300" /> 10-Tier Resilient LLM Cascade
                </div>
                <button
                  onClick={() => setShowModelModal(false)}
                  className="text-zinc-400 hover:text-white p-1 rounded-lg hover:bg-zinc-800 transition-colors">
                  <X className="w-4 h-4" />
                </button>
              </div>
              <p className="text-xs text-zinc-400 mb-3 leading-relaxed">
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
                    className="flex items-center justify-between p-2.5 rounded-xl bg-[#121214] border border-zinc-800 text-xs">
                    <div className="flex items-center gap-2.5">
                      <span className="w-5 h-5 rounded-full bg-zinc-800 text-zinc-300 border border-zinc-700 flex items-center justify-center font-bold text-[10px]">
                        {m.priority || idx + 1}
                      </span>
                      <div>
                        <div className="font-semibold text-white">
                          {m.model_name || m.name}
                        </div>
                        <div className="text-[10px] text-zinc-500 uppercase">
                          {m.provider}
                        </div>
                      </div>
                    </div>
                    <span className="text-[10px] text-zinc-200 bg-zinc-800 px-2 py-0.5 rounded border border-zinc-700 font-medium">
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
