import os
from langchain import document_loaders, text_splitter, chains, schema
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.base import Embeddings
from llm import gemini
from langchain.globals import set_debug, set_verbose

set_debug(True)
set_verbose(True)


class RedundantFilterRetriever(schema.BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma
    k: int

    def get_relevant_documents(self, query):
        emb = self.embeddings.embed_query(query)
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,
            k=self.k,
        )

    def aget_relevant_documents(self, query):
        return []


def create_db(embeddings):
    splitter = text_splitter.CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0,
    )

    loader = document_loaders.TextLoader("./section4/facts.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    db = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory="./section4/emb"
    )


def get_retriever(embeddings):
    db = Chroma(
        persist_directory="./section4/emb",
        embedding_function=embeddings,
    )

    # retriever = db.as_retriever(search_kwargs={"k": 4})
    retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db, k=4)

    return retriever


if __name__ == "__main__":
    # Get the gemini model and embeddings
    model = gemini.model
    embeddings = gemini.embeddings

    # create_db(embeddings=embeddings)
    retriever = get_retriever(embeddings=embeddings)

    chain = chains.RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="map_reduce",
    )

    result = chain.run("What is an interesting fun fact about a great scientist ?")

    print(result)
