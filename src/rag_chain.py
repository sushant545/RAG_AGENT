from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from .config import get_openai_key





def create_rag_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",   # ✅ STRING — QUOTES REQUIRED
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def rag_with_sources(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

        answer = llm.invoke(prompt)
        return {
            "answer": answer.content,
            "sources": docs,
        }

    return rag_with_sources