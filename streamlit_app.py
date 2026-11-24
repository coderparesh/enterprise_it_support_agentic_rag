import streamlit as st
import os
from pathlib import Path
from app.rag.workflow import ask
from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents
from app.core.config import get_settings

settings = get_settings()

st.set_page_config(
    page_title="Enterprise IT Support Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Clean & without any broken HTML tags)
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0a0d12;
        color: #edf2f7;
    }
    
    /* Brand Badge */
    .brand-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .16em;
        background: #183b2e;
        color: #78e5af;
        padding: 5px 10px;
        border-radius: 7px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    /* Workflow Step */
    .flow-step {
        font-size: 12.5px;
        padding: 7px 10px;
        border-left: 2px solid #3a4658;
        color: #bac4d1;
        margin: 5px 0;
        background: #121923;
        border-radius: 0 6px 6px 0;
    }
    
    .source-badge-kb {
        display: inline-block;
        background-color: rgba(68, 196, 134, 0.15);
        color: #78e5af;
        border: 1px solid rgba(68, 196, 134, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .source-badge-web {
        display: inline-block;
        background-color: rgba(250, 204, 21, 0.15);
        color: #facc15;
        border: 1px solid rgba(250, 204, 21, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .trace-box {
        font-size: 12px;
        background: #151c26;
        border: 1px solid #273244;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        color: #d1d5db;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<span class="brand-badge">FDE Project</span>', unsafe_allow_html=True)
    st.title("IT Support Copilot")
    st.caption("Private company knowledge first. External web only when evidence is weak.")

    st.divider()

    st.subheader("LangGraph Workflow")
    st.markdown('<div class="flow-step">1. Route question</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-step">2. Retrieve private KB</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-step">3. Grade evidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-step">4. Web fallback if weak</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-step">5. Rewrite & retry</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-step">6. Grounded answer</div>', unsafe_allow_html=True)

    st.divider()

    with st.expander("📄 + Add Company Document"):
        st.caption("Upload PDF, TXT, MD, or DOCX to chunk, embed, and index into Pinecone.")
        uploaded_file = st.file_uploader("Select document", type=["pdf", "txt", "md", "docx"], label_visibility="collapsed")
        admin_key = st.text_input("Admin API Key", type="password", placeholder="Enter admin key")
        
        if st.button("Index Document", use_container_width=True):
            if not uploaded_file:
                st.error("Please select a file to upload.")
            elif admin_key != settings.admin_api_key:
                st.error("Invalid Admin API key.")
            else:
                with st.spinner("Indexing document into Pinecone..."):
                    upload_dir = Path(settings.upload_dir)
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    dest = upload_dir / uploaded_file.name
                    with open(dest, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    docs = load_file(dest)
                    chunks = chunk_documents(docs)
                    ids = add_documents(chunks)
                    st.success(f"Indexed {uploaded_file.name} ({len(chunks)} chunks)")

# ----------------- MAIN LAYOUT -----------------
col_chat, col_trace = st.columns([2.2, 1], gap="medium")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "**How can I help with IT support?**\n\nTry: *“How do I connect to the company VPN?”* or ask a question that may require current external vendor information.",
            "source": "",
            "citations": [],
            "trace": []
        }
    ]

if "latest_trace" not in st.session_state:
    st.session_state.latest_trace = []
    st.session_state.latest_source = "—"

with col_chat:
    st.subheader("Enterprise Knowledge Copilot")
    st.caption("Grounded Agentic RAG • LangGraph • Pinecone • Groq • Tavily")

    # Example question buttons
    col1, col2, col3 = st.columns(3)
    preset_query = None
    if col1.button("VPN Connection Guide", use_container_width=True):
        preset_query = "How do I connect to the company VPN from home?"
    if col2.button("MFA Sharing Policy", use_container_width=True):
        preset_query = "Can IT support ask me to share my MFA code?"
    if col3.button("Teams Outage Guidance", use_container_width=True):
        preset_query = "What is the latest Microsoft Teams outage guidance?"

    # Display Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Source Tag
            if msg.get("source"):
                src = msg["source"].upper()
                badge_class = "source-badge-kb" if "KB" in src else "source-badge-web"
                st.markdown(f'<span class="{badge_class}">Source: {src}</span>', unsafe_allow_html=True)
            
            # Citations
            if msg.get("citations"):
                with st.expander("📚 Citations & Sources"):
                    for c in msg["citations"]:
                        title = c.get("title", "Document")
                        url = c.get("url")
                        if url:
                            st.markdown(f"- [{title}]({url})")
                        else:
                            st.markdown(f"- {title}")

    # Chat Input Box
    input_text = st.chat_input("Ask an IT support question...")
    user_query = preset_query or input_text

    if user_query:
        # Add User Message
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "source": "",
            "citations": [],
            "trace": []
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing question & navigating knowledge graph..."):
                result = ask(user_query)
                answer = result.get("answer", "")
                source_used = result.get("source_used", "")
                trace = result.get("trace", [])
                citations = result.get("citations", [])

                st.markdown(answer)

                if source_used:
                    src = source_used.upper()
                    badge_class = "source-badge-kb" if "KB" in src else "source-badge-web"
                    st.markdown(f'<span class="{badge_class}">Source: {src}</span>', unsafe_allow_html=True)

                if citations:
                    with st.expander("📚 Citations & Sources"):
                        for c in citations:
                            title = c.get("title", "Document")
                            url = c.get("url")
                            if url:
                                st.markdown(f"- [{title}]({url})")
                            else:
                                st.markdown(f"- {title}")

                # Update State
                st.session_state.latest_trace = trace
                st.session_state.latest_source = source_used.upper() if source_used else "—"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source": source_used,
                    "citations": citations,
                    "trace": trace
                })
                st.rerun()

# ----------------- TRACE PANEL -----------------
with col_trace:
    st.subheader("Agent Trace")
    st.caption("Watch LangGraph decisions while the request runs.")

    st.metric("Final Source", st.session_state.latest_source)

    if st.session_state.latest_trace:
        for t in st.session_state.latest_trace:
            st.markdown(f'<div class="trace-box">{t}</div>', unsafe_allow_html=True)
    else:
        st.info("No query yet. Ask a question to see the execution trace.")
