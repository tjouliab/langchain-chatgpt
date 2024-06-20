import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
llm_path = os.path.join(base_dir, "..")
sys.path.insert(0, os.path.abspath(llm_path))

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores.chroma import Chroma

from llm import gemini

if __name__ == "__main__":
    model = gemini.model
    embeddings = gemini.embeddings

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0,
    )

    facts_file_path = os.path.join(base_dir, "../facts.txt")
    loader = TextLoader(facts_file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)

    db = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory="src/facts-emb"
    )
