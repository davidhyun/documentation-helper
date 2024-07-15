import os
from dotenv import load_dotenv; load_dotenv()
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain import hub


def run_llm(query: str) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    retriever = PineconeVectorStore.from_existing_index(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)
    
    chat = ChatOpenAI(temperature=0, verbose=True)
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    
    qa = create_retrieval_chain(
        retriever=retriever.as_retriever(),
        combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    
    return new_result


if __name__ == "__main__":
    result = run_llm(query="What is RetrievalQA chain?")
    print(result)