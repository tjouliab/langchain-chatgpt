import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
llm_path = os.path.join(base_dir, "..")
sys.path.insert(0, os.path.abspath(llm_path))

from llm import gemini

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores.chroma import Chroma


# Adding context with embedding techniques
if __name__ == "__main__":
    # Init Model and Embeddings
    model = gemini.model
    embeddings = gemini.embeddings

    # Split the content of the .txt file
    loader = TextLoader("src/facts.txt")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0,
    )

    docs = loader.load_and_split(text_splitter=text_splitter)

    # Add docs to the vector db
    db = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory="src/facts-emb"
    )
