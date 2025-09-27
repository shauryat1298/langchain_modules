from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_xai import ChatXAI
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv


load_dotenv()

os.environ["XAI_API_KEY"] = os.getenv("XAI_API_KEY")

app = FastAPI(
    title="Langchain server",
    version=1.0,
    description="Simple API Server"
)

add_routes(app, ChatXAI(), path="/xai")

model = ChatXAI(model="grok-code-fast-1")


prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} in 100 words")

add_routes(app, prompt1 | model, path="/essay")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

