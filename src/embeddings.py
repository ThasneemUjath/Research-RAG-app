from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunks):
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create FAISS vector store from chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def get_retriever(vector_store):
    return vector_store.as_retriever(
        search_kwargs={"k": 5}  # fetch top 3 relevant chunks
    )