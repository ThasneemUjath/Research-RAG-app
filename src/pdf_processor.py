from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def process_pdf(uploaded_file):
    # Check file size
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
        raise ValueError("File too large! Please upload a PDF under 50MB.")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # Load PDF and extract text
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        
        # Check if PDF has extractable text
        total_text = " ".join([doc.page_content for doc in documents])
        if len(total_text.strip()) < 100:
            raise ValueError("PDF appears to be scanned or has no extractable text!")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    
    except Exception as e:
        raise e
    
    finally:
        # Always clean up temp file
        os.unlink(tmp_path)