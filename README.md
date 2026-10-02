#  Research Paper Assistant

An AI-powered web application that lets you upload any research paper (PDF) and instantly get a structured summary or ask questions about it — powered by RAG (Retrieval Augmented Generation).

 **Live Demo:** [Click here to try it](https://research-rag-app-yo5qs8rahtqxd8us733ly3.streamlit.app/)

---

## Features

-  **Smart Summarization** — Generates a structured summary covering main topic, objectives, methodology, findings and technologies
-  **Q&A Chat** — Ask any question about the paper in natural language and get accurate answers
-  **Source Citations** — Every answer shows exactly which page and section it came from
-  **Dark Theme UI** — Clean, professional interface built with Streamlit
-  **Fast Inference** — Powered by Groq's LPU hardware for ultra-fast responses

---

## Tech Stack

| Technology | Purpose |
|---|---|
|  GPT-OSS 120B (Groq) | LLM for Q&A and summarization |
|  LangChain | RAG pipeline orchestration |
|  FAISS | Vector database for similarity search |
|  HuggingFace Embeddings | Converting text chunks to embeddings |
|  PyMuPDF | PDF text extraction |
|  Streamlit | Web application framework |
|  Python | Core programming language |

---

## How It Works

```
PDF Upload
    ↓
Extract Text (PyMuPDF)
    ↓
Split into Chunks (LangChain)
    ↓
Convert to Embeddings (HuggingFace)
    ↓
Store in FAISS Vector DB
    ↓
User asks a question
    ↓
FAISS finds relevant chunks
    ↓
Groq LLM reads chunks + answers
    ↓
Answer displayed with source citations
```

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/ThasneemUjath/Research-RAG-app.git
cd Research-RAG-app
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

---

## Project Structure

```
Research-RAG-app/
├── src/
│   ├── pdf_processor.py      # PDF loading and chunking
│   ├── embeddings.py         # HuggingFace embeddings + FAISS
│   ├── qa_chain.py           # RAG Q&A chain
│   └── summarizer.py         # Summarization pipeline
├── .streamlit/
│   └── config.toml           # Dark theme config
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Example Questions to Ask

- *"What is this paper about?"*
- *"What methodology is used?"*
- *"What are the key findings?"*
- *"What technologies are used?"*
- *"What problem does this paper solve?"*
- *"Who are the authors?"*

---

## Future Improvements

- [ ] Support for multiple PDF uploads
- [ ] Export chat history as PDF
- [ ] Support for scanned PDFs using OCR
- [ ] Multi-language support
- [ ] Chat history persistence across sessions

---

