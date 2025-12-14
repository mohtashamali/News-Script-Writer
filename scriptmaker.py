from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables!")


def create_db(full_text):
    # print("🔪 Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(full_text)

    embedder = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


    db = FAISS.from_texts(chunks, embedder)
    
    return db


def generate_news(db):
    retriever = db.as_retriever(search_kwargs={"k": 6})
    

    relevant = retriever.invoke("Generate a news script")

    context = "\n\n---\n\n".join([doc.page_content for doc in relevant])

    prompt = f"""
You are a professional Indian news script writer.
Use ONLY the following extracted information to create a news script.

Information:
{context}

FORMAT:

1. Story Overview 
2. Detailed News Body (15–25 lines)
3. Key Facts (bullet points)
4. Outro: "Reporting for News Desk."

RULES:
- No hallucinations.
- Use only the extracted content.
- Clear and crisp anchor-style news writing.
- Write in a professional broadcast news tone.
"""

    # print("🤖 Generating news script with LLM...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )
    result = llm.invoke(prompt)
    return result.content