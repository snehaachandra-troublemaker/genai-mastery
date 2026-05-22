import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

""" from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist() """

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

def build_parent_child_retriever(documents):
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    
    #embeddings = LocalEmbeddings()

    vectorstore = FAISS.from_documents(
        [Document(page_content="init", metadata={})],
        embeddings
    )

    docstore = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        docstore=docstore,
    )

    retriever.add_documents(documents)
    return retriever

def query_rag(retriever, llm, query):
    parent_docs = retriever.invoke(query)

    context = "\n\n--\n".join([f"[{i}] {doc.metadata.get('policy_type', 'Unknown Policy')} | {doc.metadata.get('source_file', 'Unknown Source')} | {doc.page_content}" for i, doc in enumerate(parent_docs, 1)])

    prompt = f"""
    You are a helpful assistant.
    Use the following information to answer the question.
    {context}
    Question: {query}
    INSTRUCTIONS:
    1. Answer the question based ONLY on the provided context
    2. Be specific and cite which policy section you're referencing
    3. If the context doesn't contain the answer, say "I don't have that information in the policies provided"
    4. Keep answer concise but complete (1-2 paragraphs max)
    5. Include relevant details like timeframes, contact info, or process steps
    """

    response = llm.invoke(prompt)

    return response.content, parent_docs
    


if __name__ == "__main__":
    policy_dir = "/Users/sneha/projects/week2-langchain/policies"
    docs = load_documents(policy_dir)
    retriever = build_parent_child_retriever(docs)
    
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash", temperature = 0.3
    )
    
    while True:
        query = input("Ask: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        answer, parent_docs = query_rag(retriever, llm, query)
        print(answer)
        print("Sources: ")
        for doc in parent_docs:
            print(f"  - {doc.metadata['policy_type']} | {doc.page_content[:100]}...")
