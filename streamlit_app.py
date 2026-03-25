import streamlit
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import transformers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import pypdf
import sentence_transformers
import chromadb


def build_rag(pdf):
    loader = PyPDFLoader(pdf)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(chunks, embedding_model)

    return vectorstore

vector_database = build_rag("EU_AI_ACT_2024.pdf")

example = vector_database.similarity_search("requirements for high-risk AI systems?", k=3)
print(example)