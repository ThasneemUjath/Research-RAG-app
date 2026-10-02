from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def create_qa_chain(retriever):
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    prompt_template = PromptTemplate(
        template="""
You are a helpful research assistant. Use the following context from a research paper to answer the question clearly and accurately.
If the answer is not in the context, say "I couldn't find that in the paper."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain that also returns source documents
    def get_answer_with_sources(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        chain = (
            prompt_template
            | llm
            | StrOutputParser()
        )
        
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        return {
            "answer": answer,
            "sources": docs
        }

    return get_answer_with_sources