from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def create_summarizer():
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    
    prompt = PromptTemplate(
        template="""
You are an expert research assistant. Read the following content from a research paper and provide:

1. **Main Topic** - What is this paper about? (2-3 sentences)
2. **Key Objectives** - What does it aim to achieve? (bullet points)
3. **Methodology** - How does it work? (brief)
4. **Key Findings / Contributions** - What are the main results or contributions?
5. **Technologies Used** - List the main tools/technologies mentioned

Content:
{content}

Provide a clear, structured summary.
""",
        input_variables=["content"]
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain

def summarize_document(chunks):
    # Take chunks from beginning, middle and end for better coverage
    total = len(chunks)
    
    beginning = chunks[:4]                          # intro/abstract
    middle = chunks[total//4 : total//4 + 4]        # methodology
    end = chunks[total//2 : total//2 + 4]           # results/conclusion
    
    selected_chunks = beginning + middle + end
    content = "\n\n".join([chunk.page_content for chunk in selected_chunks])
    
    summarizer = create_summarizer()
    summary = summarizer.invoke({"content": content})
    return summary