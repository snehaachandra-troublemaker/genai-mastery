import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

load_dotenv()

def load_documents(policy_dir):
    policy_dir = Path(policy_dir)
    documents = []
    for file_path in policy_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        metadata = {'source_file': file_path.name, 'policy_type': file_path.stem.replace("_policy", "").upper() }
        documents.append(Document(page_content=content, metadata=metadata))

    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators = ["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_documents(documents)

def build_vectorstore(docs):
    # embeddings = HuggingFaceEmbeddings(
    #     model_name = "sentence-transformers/all-MiniLM-L6-v2",
    #     encode_kwargs = {'normalize_embeddings': True}  # important for cosine similarity
    # )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash", temperature = 0.3
        )
    
    return RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_kwargs = {"k": 3}), 
        return_source_documents=True
    )

if __name__ == "__main__":
    policy_dir = "/Users/sneha/projects/week2-langchain/policies"
    docs = load_documents(policy_dir)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore)
    
    while True:
        query = input("Ask: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        result = qa_chain.invoke({"query": query})
        print(result["result"])
        print("Sources: ")
        for doc in result["source_documents"]:
            print(f"  - {doc.metadata['policy_type']} | {doc.page_content[:100]}...")

    