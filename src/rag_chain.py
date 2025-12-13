from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


from .config import get_openai_key


def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    base_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 🔑 NEW: wrapper that returns sources
    def rag_with_sources(question: str):
        docs = retriever.invoke(question)
        answer = base_chain.invoke(question)

        return {
            "answer": answer,
            "sources": docs
        }

    return rag_with_sources
