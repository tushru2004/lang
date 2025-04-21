from langchain.chat_models import ChatOpenAI
import os
import pinecone
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain_core.prompts import PromptTemplate

from src.services.langchain.question import CustomRetriever

class QuestionBuilder:
    def build(self,question):
        embeddings = OpenAIEmbeddings()
        pinecone.Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV_NAME")
        )

        vector_store = Pinecone.from_existing_index(
            os.getenv("PINECONE_INDEX_NAME"), embeddings
        )
        retriever = CustomRetriever(
            embeddings=embeddings,
            pinecone=vector_store
        )
        chat = ChatOpenAI()
        prompt_template = PromptTemplate.from_template("""
        You are a helpful assistant. Use ONLY the following context to answer the question.
        If the answer is not in the context, respond with "I don't know."

        Context:
        {context}

        Question: {question}
        Answer:
        """)
        llm_chain = LLMChain(llm=chat, prompt=prompt_template)

        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )
        chain = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=stuff_chain,
            return_source_documents=True
        )

        #result = chain.run(question)

        result = chain({"query": question})

        return "{}".format(result["result"])