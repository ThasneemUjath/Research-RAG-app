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
    .main { background-color: #0f1117; }
    .stTextInput > div > div > input {
        background-color: #1e1e2e;
        color: white;
        border-radius: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 10px;
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
    .user-message b, .assistant-message b {
        color: #a78bfa !important;
    }
    .assistant-message b {
        color: #60a5fa !important;
    }
    p, div, span, li {
        color: #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("# 📄 Research Paper Assistant")
st.markdown("Upload any research paper and ask questions or get a summary instantly!")
st.divider()

with st.sidebar:
    st.markdown("## 📂 Upload Paper")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
    
    st.divider()
    st.markdown("## ℹ️ How it works")
    st.markdown("""
    1. 📤 Upload a research paper
    2. ⚙️ App processes the PDF
    3. 📝 Get an instant summary
    4. 💬 Ask any question about it
    """)
    
    st.divider()
    st.markdown("## 🛠️ Built With")
    st.markdown("""
    - 🦙 Llama 3.3 70B (Groq)
    - 🔗 LangChain
    - 🗄️ FAISS Vector DB
    - 🤗 HuggingFace Embeddings
    - 🎈 Streamlit
    """)

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
            st.success("✅ Paper ready! Ask questions or generate a summary.")
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        except Exception as e:
            st.error("❌ Something went wrong processing the PDF. Please try another file.")
            st.stop()
    tab1, tab2 = st.tabs(["📝 Summary", "💬 Chat"])

    with tab1:
        st.subheader("📝 Paper Summary")
        st.markdown("Get a structured summary covering the main topic, objectives, methodology, findings and technologies.")
        
        if st.button("✨ Generate Summary", use_container_width=True):
            with st.spinner("📖 Reading and summarizing the paper..."):
                summary = summarize_document(st.session_state.chunks)
            st.markdown(summary)

    with tab2:
        col1, col2 = st.columns([4, 1])
        with col1:  
            st.subheader("💬 Chat with the Paper")
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

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
                
                # Show sources
                if msg.get("sources"):
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(f"**Source {i+1} — Page {source.metadata.get('page', 'N/A') + 1}:**")
                            st.caption(source.page_content[:300] + "...")
                            st.divider()

        question = st.chat_input("Ask anything about the paper...")

        if question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.spinner("🤔 Thinking..."):
                result = st.session_state.qa_chain(question)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })

            st.rerun()

else:
    st.markdown("""
    <div style='text-align: center; padding: 4rem; color: #666;'>
        <h2>👈 Upload a research paper to get started</h2>
        <p>Supports any PDF research paper or document</p>
    </div>
    """, unsafe_allow_html=True)