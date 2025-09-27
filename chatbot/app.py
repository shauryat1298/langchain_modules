import os
from dotenv import load_dotenv

import streamlit as st
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Ensure API keys are available
os.environ["XAI_API_KEY"] = os.getenv("XAI_API_KEY", "")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer under 100 words!"),
        ("user", "Question: {question}")
    ]
)

# Streamlit app
st.title("LangChain Demo with Grok Fast")
input_text = st.text_input("Search for the topic you want!")

# XAI model
model = ChatXAI(model="grok-code-fast-1")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
