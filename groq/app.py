import os
from dotenv import load_dotenv
import time

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

load_dotenv()

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/langsmith/home")
    st.session_state.docs =  st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector_db = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Chat Groq Demo")
llm = ChatGroq(model = "llama-3.1-8b-instant", temperature=0)
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Context: {context}
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your question")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time: ", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("___________________________")



