import streamlit as st
from dotenv import load_dotenv
import os
from src.pdf_processor import process_pdf
from src.embeddings import create_vector_store, get_retriever
from src.qa_chain import create_qa_chain
from src.summarizer import summarize_document

load_dotenv()

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
    <style>
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 12px;
        line-height: 1.6;
    }
    .user-message {
        background-color: #1e1e2e;
        border-left: 4px solid #7c3aed;
        color: #ffffff !important;
    }
    .assistant-message {
        background-color: #162032;
        border-left: 4px solid #2563eb;
        color: #ffffff !important;
    }
    .user-message b { color: #a78bfa !important; }
    .assistant-message b { color: #60a5fa !important; }
    p, div, span, li { color: #e2e8f0 !important; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 📄 Research Paper Assistant")
    st.markdown("Upload any research paper PDF — get an instant summary or ask questions about it.")

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📂 Upload Paper")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        file_size = round(uploaded_file.size / (1024 * 1024), 2)
        st.caption(f"📦 Size: {file_size} MB")

    st.divider()

    st.markdown("## 🛠️ Built With")
    st.markdown("""
    - 🤖 GPT-OSS 120B (Groq)
    - 🔗 LangChain
    - 🗄️ FAISS Vector DB
    - 🤗 HuggingFace Embeddings
    - 🎈 Streamlit
    """)

    st.divider()
    st.markdown("## 👨‍💻 About")
    st.markdown("""
    This app uses **RAG (Retrieval Augmented Generation)** to answer questions from any research paper accurately.
    """)

# --- Main Content ---
if uploaded_file:
    if "chunks" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        try:
            with st.spinner("⚙️ Processing PDF..."):
                chunks = process_pdf(uploaded_file)
                vector_store = create_vector_store(chunks)
                retriever = get_retriever(vector_store)
                qa_chain = create_qa_chain(retriever)

                st.session_state.chunks = chunks
                st.session_state.retriever = retriever
                st.session_state.qa_chain = qa_chain
                st.session_state.file_name = uploaded_file.name
                st.session_state.chat_history = []

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 File", uploaded_file.name[:20] + "...")
            with col2:
                st.metric("🧩 Chunks", len(chunks))
            with col3:
                st.metric("📦 Size", f"{round(uploaded_file.size / (1024*1024), 2)} MB")

        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        except Exception as e:
            st.error("❌ Something went wrong. Please try another PDF.")
            st.stop()

    tab1, tab2 = st.tabs(["📝 Summary", "💬 Chat"])

    # --- Summary Tab ---
    with tab1:
        st.subheader("📝 Paper Summary")
        st.markdown("Generates a structured summary covering the main topic, objectives, methodology, findings and technologies.")

        if st.button("✨ Generate Summary", use_container_width=True):
            with st.spinner("📖 Reading and summarizing..."):
                summary = summarize_document(st.session_state.chunks)
            st.markdown(summary)

    # --- Chat Tab ---
    with tab2:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("💬 Chat with the Paper")
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Suggested questions
        if not st.session_state.chat_history:
            st.markdown("**💡 Try asking:**")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("What is this paper about?", use_container_width=True):
                    st.session_state.suggested = "What is this paper about?"
            with c2:
                if st.button("What methodology is used?", use_container_width=True):
                    st.session_state.suggested = "What methodology is used?"
            with c3:
                if st.button("What are the key findings?", use_container_width=True):
                    st.session_state.suggested = "What are the key findings?"

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    🧑 <b>You:</b> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    🤖 <b>Assistant:</b> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)

                if msg.get("sources"):
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(f"**Source {i+1} — Page {source.metadata.get('page', 'N/A') + 1}:**")
                            st.caption(source.page_content[:300] + "...")
                            st.divider()

        # Handle suggested question
        if "suggested" in st.session_state:
            question = st.session_state.suggested
            del st.session_state.suggested
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.qa_chain(question)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })
            st.rerun()

        # Chat input
        question = st.chat_input("Ask anything about the paper...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.qa_chain(question)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })
            st.rerun()

else:
    # Landing page
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>👈 Upload a research paper to get started</h2>
        <p style='color: #666;'>Supports any PDF research paper or document</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📝 **Smart Summarization**\n\nGet structured summaries covering objectives, methodology and findings instantly.")
    with c2:
        st.info("💬 **Q&A Chat**\n\nAsk any question about the paper and get accurate answers with source references.")
    with c3:
        st.info("📚 **Source Citations**\n\nEvery answer shows exactly which part of the paper it came from.")