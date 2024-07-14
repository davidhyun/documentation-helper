import os
from dotenv import load_dotenv; load_dotenv()
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")
    
    for doc in documents:
        old_path = doc.metadata['source']
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
    
    print(f"Going to add {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(documents, embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("***** Added to Pinecone vectorstore vectors *****")    
    
    
if __name__ == "__main__":
    ingest_docs()