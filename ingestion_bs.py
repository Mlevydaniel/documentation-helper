from dotenv import load_dotenv

load_dotenv()

import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from consts import INDEX_NAME

# from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import BSHTMLLoader

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def load_html_files(directory_path):
    loaded_files = []

    # Traverse the directory tree
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.html'):
                file_path = os.path.join(root, file)
                loader = BSHTMLLoader(file_path)
                structured_data = loader.load()
                loaded_files.extend(structured_data)

    return loaded_files


def extract_text(structured_data):
    text_content = ""
    for element in structured_data:
        if hasattr(element, 'page_content'):
            text_content += element.page_content + "\n"
    return text_content


def ingest_docs():
    directory_path = 'langchain-docs/python.langchain.com/'
    loaded_html_files = load_html_files(directory_path)

    # raw_documents = [extract_text(file_data) for file_data in loaded_html_files]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(loaded_html_files)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=INDEX_NAME
    )

    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
