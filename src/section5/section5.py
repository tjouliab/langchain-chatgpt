import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
llm_path = os.path.join(base_dir, "..")
sys.path.insert(0, os.path.abspath(llm_path))

from llm import gemini

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain.globals import set_debug, set_verbose

set_debug(True)
set_verbose(True)


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str) -> None:
        emb = self.embeddings.embed_query(query)
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8
        )

    async def aget_relevant_documents(self) -> None:
        return []


if __name__ == "__main__":
    model = gemini.model
    embeddings = gemini.embeddings

    db = Chroma(embedding_function=embeddings, persist_directory="src/facts-emb")

    retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)

    # "stuff", "map_reduce", "map_rerank", "refine"
    chain = RetrievalQA.from_chain_type(
        llm=model, retriever=retriever, chain_type="refine"
    )

    result = chain.run("What is an interesting fact about a famous scientist ?")
    print(result)
