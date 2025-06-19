import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_PATH = 'company_docs'

# Load all PDF and text files from the docs folder
def load_documents():
    documents = []
    for filename in os.listdir(DOCS_PATH):
        filepath = os.path.join(DOCS_PATH, filename)
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.lower().endswith('.txt'):
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    return documents

# Split documents into chunks for retrieval
def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents) 