"""Streamlit Financial RAG Dashboard — interactive UI for SEC 10-K analysis."""

import json
import uuid
from pathlib import Path
from typing import Any, Iterator

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
SUPPORTED_TICKERS = ["AAPL", "MSFT", "NVDA"]
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".html", ".htm", ".txt", ".sgml"})

# ---------------------------------------------------------------------------
# API Client Wrapper
# ---------------------------------------------------------------------------


def check_health(base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    """Probe the /health endpoint and return service status."""
    resp = httpx.get(f"{base_url}/health", timeout=5)
    resp.raise_for_status()
    return resp.json()


def send_query(
    query: str,
    ticker: str | None = None,
    fiscal_year: str | None = None,
    session_id: str | None = None,
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    """Send a chat query (sync) and return the full response dict."""
    payload: dict[str, Any] = {"user_query": query}
    if ticker:
        payload["ticker"] = ticker
    if fiscal_year:
        payload["fiscal_year"] = fiscal_year
    if session_id:
        payload["session_id"] = session_id

    resp = httpx.post(f"{base_url}/api/v1/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def stream_query(
    query: str,
    ticker: str | None = None,
    fiscal_year: str | None = None,
    session_id: str | None = None,
    base_url: str = DEFAULT_API_URL,
) -> Iterator[str]:
    """Stream tokens from the SSE chat endpoint.

    Yields individual text tokens as they are generated server-side.
    """
    payload: dict[str, Any] = {"user_query": query}
    if ticker:
        payload["ticker"] = ticker
    if fiscal_year:
        payload["fiscal_year"] = fiscal_year
    if session_id:
        payload["session_id"] = session_id

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{base_url}/api/v1/chat/stream", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: done"):
                    break
                if line.startswith("data: "):
                    token = line[len("data: "):]
                    if token:
                        yield token


def upload_document(
    file_bytes: bytes,
    filename: str,
    ticker: str = "UNKNOWN",
    fiscal_year: str = "UNKNOWN",
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    """Upload a document and trigger background ingestion."""
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    params = {"ticker": ticker, "fiscal_year": fiscal_year}
    files = {"file": (filename, file_bytes)}
    resp = httpx.post(
        f"{base_url}/api/v1/documents/upload",
        params=params,
        files=files,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Financial Theme
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp { background-color: #0F172A; }
    section[data-testid="stSidebar"] > div { background-color: #1E293B; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] * { color: #F1F5F9 !important; }
    [data-testid="stSidebar"] label { color: #94A3B8 !important; font-weight: 600; font-size: 0.85rem; }
    [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stTextInput label { color: #94A3B8 !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div { background-color: #0F172A; border: 1px solid #475569; color: #F1F5F9; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover { border-color: #10B981; }
    [data-testid="stSidebar"] .stTextInput input { background-color: #0F172A; border: 1px solid #475569; color: #F1F5F9; }
    [data-testid="stSidebar"] .stTextInput input:focus { border-color: #10B981; box-shadow: 0 0 0 1px #10B981; }
    [data-testid="stSidebar"] .stFileUploader section { background-color: #0F172A; border: 1px dashed #475569; color: #94A3B8; }
    [data-testid="stSidebar"] .stFileUploader section:hover { border-color: #10B981; }
    [data-testid="stSidebar"] .st-emotion-cache-1mi2ry1 { color: #94A3B8 !important; }
    [data-testid="stSidebar"] div.stCaption, [data-testid="stSidebar"] caption { color: #64748B !important; }
    [data-testid="stSidebar"] hr { border-color: #334155 !important; }
    [data-testid="stSidebar"] .stButton button { background-color: #10B981; color: #fff !important; border: none; font-weight: 600; }
    [data-testid="stSidebar"] .stButton button:hover { background-color: #059669; }
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p { color: #E2E8F0 !important; }
    div.card {
        background: #1E293B; border: 1px solid #334155; border-radius: 12px;
        padding: 1.25rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); margin-bottom: 1rem;
    }
    span.badge {
        display: inline-block; padding: 0.2em 0.6em; font-size: 0.75rem;
        font-weight: 600; border-radius: 9999px; margin-left: 0.4em;
    }
    span.badge-ok { background: #10B981; color: #fff; }
    span.badge-warn { background: #F59E0B; color: #1E293B; }
    span.badge-err { background: #EF4444; color: #fff; }
    [data-testid="stMetricValue"] { color: #10B981; }
    .st-emotion-cache-1c7y2kd { background: #1E293B; border-radius: 12px; }
    .streamlit-expanderHeader { background: #1E293B; border: 1px solid #334155; border-radius: 8px; }
    .footer-bar { font-size: 0.75rem; color: #64748B; margin-top: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
if "api_base" not in st.session_state:
    st.session_state.api_base = DEFAULT_API_URL

_APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="card"><h3 style="margin:0;color:#F1F5F9;">⚙️ System Health</h3></div>', unsafe_allow_html=True)

    health: dict[str, Any] | None = None
    try:
        health = check_health(st.session_state.api_base)
    except Exception:
        pass

    cols = st.columns(3)
    for i, (svc, label) in enumerate([("mongodb", "MongoDB"), ("qdrant", "Qdrant"), ("redis", "Redis")]):
        with cols[i]:
            status = health["services"][svc] if health else "unknown"
            ok = status == "ok"
            badge_class = "badge-ok" if ok else "badge-err"
            badge_text = "✓" if ok else "✗"
            st.markdown(
                f'<div style="text-align:center;color:#94A3B8;font-size:0.8rem;">'
                f'{label}<br><span style="font-size:1.2rem;">'
                f'<span class="badge {badge_class}">{badge_text}</span></span></div>',
                unsafe_allow_html=True,
            )

    # Warm-up status indicator
    warmup_completed = health.get("warmup_completed", False) if health else False
    wu_badge_class = "badge-ok" if warmup_completed else "badge-warn"
    wu_text = "✓" if warmup_completed else "⟳"
    st.markdown(
        f'<div style="text-align:center;color:#94A3B8;font-size:0.8rem;margin-top:0.3rem;">'
        f'Warm-up<br><span style="font-size:1.2rem;">'
        f'<span class="badge {wu_badge_class}">{wu_text}</span></span></div>',
        unsafe_allow_html=True,
    )

    # Informational hint when warm-up is still pending
    if health and not warmup_completed:
        st.caption(
            "⏳ First query may be slower — "
            "models are still loading on the server."
        )

    st.divider()

    # --- Document Upload ---
    st.markdown('<div class="card"><h3 style="margin:0;color:#F1F5F9;">📄 Upload Document</h3></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "html", "htm", "txt", "sgml"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        upload_bytes = uploaded_file.read()
        upload_ticker = st.selectbox("Ticker", SUPPORTED_TICKERS, key="upload_ticker")
        upload_year = st.text_input("Fiscal Year", "2025", key="upload_year")
        if st.button("📤 Upload & Index", type="primary", use_container_width=True):
            with st.spinner("Uploading and indexing..."):
                try:
                    result = upload_document(
                        upload_bytes, uploaded_file.name,
                        upload_ticker, upload_year,
                        st.session_state.api_base,
                    )
                    st.success(
                        f"✅ Indexed: {result.get('chunks_created', 0)} chunks "
                        f"(Mongo: {result.get('mongo_count', 0)}, "
                        f"Qdrant: {result.get('qdrant_count', 0)}) "
                        f"in {result.get('elapsed_seconds', 0):.1f}s"
                    )
                except Exception as exc:
                    st.error(f"Upload failed: {exc}")

    st.divider()

    # --- Metadata Filters ---
    st.markdown('<div class="card"><h3 style="margin:0;color:#F1F5F9;">🔍 Filters</h3></div>', unsafe_allow_html=True)

    filter_ticker = st.selectbox(
        "Ticker",
        [None] + SUPPORTED_TICKERS,
        format_func=lambda x: "All" if x is None else x,
    )
    filter_year = st.text_input("Fiscal Year", value="", placeholder="e.g. 2025")

    st.caption(f"Session: `{st.session_state.session_id}`")

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

st.markdown(
    f'<h1 style="color:#F1F5F9;margin-bottom:0.25rem;">'
    f'📊 Financial RAG — AI SEC Filing & Report Analyzer</h1>',
    unsafe_allow_html=True,
)

overall_status = health.get("status", "unknown") if health else "unreachable"
status_color = {"ok": "#10B981", "degraded": "#F59E0B", "unreachable": "#EF4444"}
warmup_completed = health.get("warmup_completed", False) if health else False
warmup_suffix = " | Warm-up ✓" if warmup_completed else " | Warm-up ⟳"
st.markdown(
    f'<div style="margin-bottom:1.5rem;">'
    f'<span style="background:{status_color.get(overall_status, "#EF4444")};'
    f'color:#fff;padding:0.2em 0.8em;border-radius:9999px;font-size:0.8rem;">'
    f'● {overall_status}{warmup_suffix}</span>'
    f'<span style="color:#64748B;margin-left:1rem;font-size:0.8rem;">v{_APP_VERSION}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            sources = meta.get("sources", [])
            if sources:
                with st.expander("📌 Extracted Financial Sources & Citations"):
                    for src in sources:
                        st.markdown(
                            f"- **{src.get('ticker', '?')}** | "
                            f"Year: {src.get('fiscal_year', '?')} | "
                            f"Section: {src.get('section', '?')} | "
                            f"Score: `{src.get('score', 0):.3f}`"
                        )
                        snippet = src.get("text_snippet", "")
                        if snippet:
                            st.code(snippet[:200], language="text")
            perf = meta.get("performance", {})
            st.markdown(
                f'<div class="footer-bar">'
                f'⚡ {perf.get("execution_time_ms", 0):.0f}ms | '
                f'🤖 {perf.get("model_used", "unknown")} | '
                f'💾 {"HIT" if perf.get("cache_hit") else "MISS"}'
                f'</div>',
                unsafe_allow_html=True,
            )

# Chat input
if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            token_stream = stream_query(
                query=prompt,
                ticker=filter_ticker,
                fiscal_year=filter_year or None,
                session_id=st.session_state.session_id,
                base_url=st.session_state.api_base,
            )
            for token in token_stream:
                full_response += token
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

            # Retrieve full metadata via sync call (cache should be warm)
            result = send_query(
                query=prompt,
                ticker=filter_ticker,
                fiscal_year=filter_year or None,
                session_id=st.session_state.session_id,
                base_url=st.session_state.api_base,
            )

            meta: dict[str, Any] = {
                "sources": result.get("sources", []),
                "performance": {
                    "execution_time_ms": result.get("execution_time_ms", 0),
                    "model_used": result.get("model_used", "unknown"),
                    "cache_hit": result.get("cache_hit", False),
                },
            }

            sources = meta["sources"]
            if sources:
                with st.expander("📌 Extracted Financial Sources & Citations"):
                    for src in sources:
                        st.markdown(
                            f"- **{src.get('ticker', '?')}** | "
                            f"Year: {src.get('fiscal_year', '?')} | "
                            f"Section: {src.get('section', '?')} | "
                            f"Score: `{src.get('score', 0):.3f}`"
                        )
                        snippet = src.get("text_snippet", "")
                        if snippet:
                            st.code(snippet[:200], language="text")

            perf = meta["performance"]
            st.markdown(
                f'<div class="footer-bar">'
                f'⚡ {perf.get("execution_time_ms", 0):.0f}ms | '
                f'🤖 {perf.get("model_used", "unknown")} | '
                f'💾 {"HIT" if perf.get("cache_hit") else "MISS"}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metadata": meta,
            })

        except Exception as exc:
            placeholder.markdown(f"❌ **Error:** {exc}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {exc}",
                "metadata": {"sources": [], "performance": {}},
            })
