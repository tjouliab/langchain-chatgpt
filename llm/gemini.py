import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME")

model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=API_KEY,
)
