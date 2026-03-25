import streamlit as st
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

@st.cache_resource
def build_rag(pdf):
    loader = PyPDFLoader(pdf)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(chunks, embedding_model)

    return vectorstore

def get_llm_reply(query):
    vector_database = build_rag("EU_AI_ACT_2024.pdf")

    # example = vector_database.similarity_search("requirements for high-risk AI systems?", k=3)
    # print(example[0].page_content)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=st.secrets["GEMINI_API_KEY"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_database.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    response = qa_chain.invoke({"query": query})
    return response["result"]


st.set_page_config(page_title="RAG EU Regulation ChatBot", layout="wide")

st.title("RAG EU Regulation ChatBot")
st.write("EU AI Act chatbot")

user_query = st.text_input("Enter Your Message Here", placeholder="e.g. What are the transparency obligations for AI providers? ")

if st.button("Send"):
    if user_query:
        st.write(f"Getting a response from AI from the database...")
        response = get_llm_reply(user_query)
        st.write(f"{response}")
    else:
        st.warning("Please enter a message")
