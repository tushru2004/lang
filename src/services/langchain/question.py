from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever
from langchain.schema import Document
from dotenv import load_dotenv
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun

load_dotenv()

class CustomRetriever(BaseRetriever):
    embeddings: OpenAIEmbeddings
    pinecone: Pinecone
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # calculate embeddings for query
        emb = self.embeddings.embed_query(query)

        docs = self.pinecone.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,
        )
        print("Docs that matched the query ", docs)
        return docs

    async def aget_relevant_documents(self):
        pass
