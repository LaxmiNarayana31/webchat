"""
Custom CSS styles for WebChat Streamlit UI.
Features a modern, glassmorphic dark theme with sleek typography and responsive cards.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* App Background */
.stApp {
    background: #09090b !important;
}

/* Main container comfortable reading width */
.main .block-container,
[data-testid="stAppViewBlockContainer"] {
    max-width: 680px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    margin: 0 auto !important;
}

/* Header & App Bars */
header[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
    height: 0 !important;
}

/* Streamlit Form & Instruction Cleanup */
div[data-testid="InputInstructions"],
[data-testid="InputInstructions"],
div[data-testid="stFormInstructions"],
.stTextInput [data-testid="InputInstructions"],
div[data-baseweb="input"] [data-testid="InputInstructions"],
div[data-testid="stWidgetLabel"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
    width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    position: absolute !important;
}

/* Column Alignment */
div[data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

/* Forms */
div[data-testid="stForm"] {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ============================================================
   HERO LANDING STYLES
   ============================================================ */
.hero-landing-wrapper {
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 8px 10px 10px 10px !important;
    margin: 0 auto 4px auto !important;
    width: 100% !important;
}

.hero-landing-wrapper h1,
.hero-landing-wrapper p,
.hero-landing-wrapper div,
[data-testid="stMarkdownContainer"] .hero-landing-wrapper,
[data-testid="stMarkdownContainer"] .hero-heading-gradient,
[data-testid="stMarkdownContainer"] .hero-subtitle-text {
    text-align: center !important;
}

.hero-super-badge {
    display: inline-block;
    padding: 4px 12px;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.12), rgba(56, 189, 248, 0.08));
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 9999px;
    color: #c4b5fd;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 8px;
}

.hero-heading-gradient {
    font-size: 26px !important;
    font-weight: 800 !important;
    text-align: center !important;
    background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 70%, #94a3b8 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 auto 6px auto !important;
    letter-spacing: -0.02em !important;
    display: block !important;
    width: 100% !important;
}

.hero-subtitle-text {
    font-size: 13px !important;
    color: #94a3b8 !important;
    max-width: 620px !important;
    width: 100% !important;
    margin: 0 auto 10px auto !important;
    line-height: 1.45 !important;
    text-align: center !important;
    display: block !important;
}

/* ============================================================
   ULTRA-PREMIUM URL INPUT BAR (STREAMLIT)
   ============================================================ */
div[data-testid="stTextInput"] {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stTextInput"] div[data-testid="stTextInputRootElement"] {
    margin: 0 !important;
    padding: 0 !important;
}

div[data-testid="stTextInput"] div[data-baseweb="input"] {
    height: 48px !important;
    min-height: 48px !important;
    max-height: 48px !important;
    box-sizing: border-box !important;
    border-radius: 10px !important;
    background: rgba(14, 16, 24, 0.85) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.14) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.25s ease !important;
    display: flex !important;
    align-items: center !important;
}

div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25), 0 8px 25px rgba(0, 0, 0, 0.5) !important;
    background: rgba(18, 20, 32, 0.95) !important;
}

div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    height: 100% !important;
    background: transparent !important;
    display: flex !important;
    align-items: center !important;
    padding: 0 !important;
}

div[data-testid="stTextInput"] input {
    height: 44px !important;
    line-height: 44px !important;
    font-size: 13.5px !important;
    color: #f8fafc !important;
    padding: 0 16px !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
}

div[data-testid="stTextInput"] input::placeholder {
    color: #64748b !important;
    font-size: 13px !important;
}

/* Text Area (URL Ingestion Box) */
div[data-testid="stTextArea"] {
    width: 100% !important;
    margin: 0 0 8px 0 !important;
    padding: 0 !important;
}

div[data-testid="stTextArea"] div[data-baseweb="textarea"] {
    min-height: 88px !important;
    height: 88px !important;
    box-sizing: border-box !important;
    border-radius: 12px !important;
    background: rgba(8, 10, 15, 0.75) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.25s ease !important;
    display: flex !important;
}

div[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.25), inset 0 2px 4px rgba(0, 0, 0, 0.5) !important;
    background: rgba(10, 13, 20, 0.95) !important;
}

div[data-testid="stTextArea"] textarea {
    min-height: 80px !important;
    height: 80px !important;
    font-size: 13px !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    color: #f1f5f9 !important;
    padding: 10px 14px !important;
    line-height: 1.45 !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    resize: none !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #475569 !important;
    font-size: 12.5px !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

/* Submit Button */
div[data-testid="stFormSubmitButton"] {
    display: flex !important;
    width: 100% !important;
}

div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
    border: none !important;
    height: 38px !important;
    min-height: 38px !important;
    width: 100% !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    letter-spacing: 0.01em !important;
    white-space: nowrap !important;
    padding: 0 16px !important;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.35) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.55) !important;
    background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%) !important;
}

div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(0) !important;
}

/* Ingestion Card */
.ingestion-card {
    background: rgba(13, 16, 23, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 12px 18px 2px 18px;
    box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.8), 0 8px 30px -10px rgba(14, 165, 233, 0.15);
    backdrop-filter: blur(20px);
    width: 100%;
    margin-bottom: 0px;
}

.ingestion-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.ingestion-card-title {
    font-size: 12.5px;
    font-weight: 600;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Quick Try Sample Buttons */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 9999px !important;
    font-size: 11.5px !important;
    font-weight: 500 !important;
    color: #cbd5e1 !important;
    height: 30px !important;
    padding: 0 10px !important;
    margin-top: 2px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
    background: rgba(139, 92, 246, 0.15) !important;
    border-color: rgba(139, 92, 246, 0.4) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Capabilities Grid */
.capabilities-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
}

.cap-card {
    background: rgba(18, 20, 30, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 12px 14px;
    transition: all 0.25s ease;
    backdrop-filter: blur(10px);
}

.cap-card:hover {
    background: rgba(24, 26, 40, 0.8);
    border-color: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.cap-icon {
    margin-bottom: 4px;
}

.cap-title {
    font-family: 'Outfit', sans-serif;
    font-size: 12.5px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 2px;
}

.cap-desc {
    font-size: 11px;
    color: #94a3b8;
    line-height: 1.4;
}

/* ============================================================
   SIDEBAR STYLES
   ============================================================ */
[data-testid="stSidebar"] {
    background: #0d0f17 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

.sidebar-section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.sidebar-count-badge {
    display: none !important;
    background: rgba(255, 255, 255, 0.08);
    padding: 2px 7px;
    border-radius: 9999px;
    font-size: 11px;
}

/* New Chat / Ingest buttons in Sidebar */
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    text-align: center !important;
    justify-content: center !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 8px 12px !important;
    line-height: 1.4 !important;
}

/* Sidebar Session Delete Button: compact, borderless trash icon */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stButton"] > button,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button {
    height: 28px !important;
    min-height: 28px !important;
    max-height: 28px !important;
    width: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    padding: 0 !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: visible !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: #64748b !important;
    opacity: 0.7 !important;
    margin: 0 auto !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

/* Material icon inside the delete button */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stButton"] > button span,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button span {
    font-size: 16px !important;
    color: #64748b !important;
}

/* Hover state */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stButton"] > button:hover,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button:hover {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ef4444 !important;
    opacity: 1 !important;
    transform: scale(1.1) !important;
}

[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stButton"] > button:hover span,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child div[data-testid="stButton"] > button:hover span {
    color: #ef4444 !important;
}

/* Center the delete column container */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Sidebar User Account Card */
.sidebar-user-card {
    background: rgba(18, 22, 34, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 14px;
    margin-top: 14px;
}

.sidebar-user-card.guest-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.65) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
}

/* Sidebar Text Input inside email form */
[data-testid="stSidebar"] div[data-testid="stTextInput"] {
    margin-bottom: 6px !important;
    height: auto !important;
}

[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-testid="stTextInputRootElement"] {
    height: 42px !important;
}

[data-testid="stSidebar"] div[data-testid="stTextInput"] div[data-baseweb="input"] {
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
    height: 40px !important;
    line-height: 40px !important;
    font-size: 13px !important;
    padding: 0 12px !important;
}

[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] > button {
    height: 40px !important;
    font-size: 13px !important;
    border-radius: 10px !important;
}

/* ============================================================
   ACTIVE DOCUMENT BANNER
   ============================================================ */
.active-doc-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(18, 22, 36, 0.75);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 14px;
    padding: 12px 18px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

.active-doc-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.active-doc-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
}

.active-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    color: #10b981;
    letter-spacing: 0.05em;
}

.active-doc-title {
    font-size: 14px;
    font-weight: 600;
    color: #f1f5f9;
}

.active-doc-url {
    font-size: 12px;
    color: #64748b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 400px;
}

/* Badges & Pills */
.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.pill-success {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.3);
}

/* Chat Messages */
.stChatMessage {
    background: rgba(15, 18, 28, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
}

/* Chat Input Box \u0026 Bottom Bar */
div[data-testid="stBottom"] {
    background: linear-gradient(to top, #09090b 60%, transparent 100%) !important;
    padding-bottom: 12px !important;
    padding-top: 20px !important;
}

div[data-testid="stBottom"] > div,
div[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}

div[data-testid="stChatInput"] {
    max-width: 960px !important;
    margin: 0 auto !important;
    border-radius: 16px !important;
    background: rgba(14, 16, 24, 0.95) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.12) !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 -4px 30px rgba(0, 0, 0, 0.5), 0 10px 40px rgba(0, 0, 0, 0.6) !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25), 0 -4px 30px rgba(0, 0, 0, 0.5) !important;
}

div[data-testid="stChatInput"] textarea {
    font-size: 14.5px !important;
    color: #f8fafc !important;
    min-height: 48px !important;
    padding-top: 14px !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #71717a !important;
    font-size: 14px !important;
}

div[data-testid="stChatInput"] button {
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
    align-self: flex-end !important;
    margin-bottom: 8px !important;
    margin-right: 4px !important;
}

div[data-testid="stChatInput"] button:hover {
    background: rgba(139, 92, 246, 0.25) !important;
    color: #a78bfa !important;
}

/* Citations Box */
.citation-box {
    background: rgba(15, 23, 42, 0.6);
    border-left: 3px solid #8b5cf6;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 12.5px;
    color: #94a3b8;
}

/* Multimodal Diagram / Markdown Images in Chat */
.stChatMessage img {
    max-width: 480px !important;
    width: 100% !important;
    height: auto !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    margin: 10px auto !important;
    display: block !important;
    background: #0f111a !important;
}

/* Knowledge Base Ready Welcome Box */
.kb-ready-box {
    text-align: center;
    padding: 28px 16px 20px 16px;
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.kb-ready-title {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
}

.kb-ready-desc {
    font-size: 13px;
    color: #94a3b8;
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.5;
}
</style>
"""
