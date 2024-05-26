from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# from langchain.chains import create_retrieval_chain

from langchain_pinecone import PineconeVectorStore
from typing import List, Any, Tuple

# from consts import INDEX_NAME

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Tuple[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0)

    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True
    # )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    qa = run_llm("What is Langchain?")
    print(qa['answer'])
