from langchain_openai import ChatOpenAI
import os


def create_rag_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def run(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are a helpful assistant.
Answer the question using the context.

Context:
{context}

Question:
{question}
"""

        answer = llm.invoke(prompt).content

        return {
            "answer": answer,
            "candidate_docs": docs  # 👈 pages only, no highlighting yet
        }

    return run
