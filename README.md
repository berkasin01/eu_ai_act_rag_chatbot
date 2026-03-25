# EU AI Act RAG Chatbot
A Retrieval Augmented Generation (RAG) chatbot that answers questions about the EU Artificial Intelligence Act (Regulation EU 2024/1689) using LangChain, ChromaDB, and Google Gemini.
[LIVE APP LINK](https://euaiactragchatbot-26kayzat3vdl3x9jl6appg8.streamlit.app/)

## What It Does
The EU AI Act is a 100+ page legal document covering how AI systems are regulated across Europe. Instead of reading the entire thing, you ask the chatbot a question and it finds the relevant sections and gives you a clear answer grounded in the actual legislation.

## Demo

<img width="1862" height="535" alt="running_app_v001" src="https://github.com/user-attachments/assets/223d67be-3560-430e-abdd-883a5b1ee7b3" />

<img width="1852" height="543" alt="running_app_v002" src="https://github.com/user-attachments/assets/3c26a495-81f6-4376-90aa-ccaac24bee2b" />

## How It Works

The EU AI Act PDF is loaded and split into chunks using LangChain's RecursiveCharacterTextSplitter
Each chunk is converted into a 384-dimensional vector embedding using the all-MiniLM-L6-v2 sentence transformer model
Embeddings are stored in a ChromaDB vector database for fast similarity search
When a user asks a question, the question is embedded using the same model and compared against the stored chunks
The 3 most relevant chunks are retrieved and sent to Google Gemini alongside the question
Gemini generates an answer based only on the retrieved context from the actual document

## Example Queries

"What AI practices are prohibited under the EU AI Act?"
"What are the requirements for high-risk AI systems?"
"How does the EU AI Act define an AI system?"
"What penalties can companies face for violating the AI Act?"
"What are the transparency obligations for AI providers?"

## Tech Stack

Python
LangChain (document loading, text splitting, retrieval chain)
ChromaDB (vector database)
HuggingFace sentence-transformers / all-MiniLM-L6-v2 (embeddings)
Google Gemini 2.5 Flash (LLM)
Streamlit (UI)

## Setup

Clone the repo
Install dependencies: pip install langchain langchain-community langchain-google-genai chromadb sentence-transformers pypdf
Download the EU AI Act PDF from EUR-Lex and place it in the project folder
Add your Google Gemini API key (get one free at aistudio.google.com)
Run the Streamlit app: streamlit run streamlit_app.py

Why This Project
RAG is one of the most widely used patterns in enterprise AI. This project demonstrates the full pipeline: document ingestion, chunking, embedding, vector storage, semantic search, and LLM-powered answer generation. The EU AI Act was chosen because it is directly relevant to AI governance and compliance, which are growing areas of focus for companies deploying AI systems.
