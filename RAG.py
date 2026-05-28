import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
print("Current working directory:", os.getcwd())

API_KEY = os.getenv("GROQ_API_KEY")
PDF_PATH = "LLM.pdf"

client = Groq(api_key=API_KEY)

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# split document
splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

chunks = splitter.split_documents(documents)

# embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vector
db = FAISS.from_documents(chunks, embeddings)

print("RAG system ready")

# user input question
while True:
    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    # LLM response
    prompt = f"""
You are an AI assistant. Answer based ONLY on the context below.

Context:
{context}

Question:
{query}

If the answer is not in the context, say "I don't know based on the document."
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    print("\nAnswer:\n", answer)