import sys
import os
import streamlit as st

st.write("DEBUG → OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR) 

from src.pdf_snapshot import generate_highlight_snapshot

# -------------------------------------------------------------------
# Imports from src
# -------------------------------------------------------------------
from src.loader import load_and_split_pdf
from src.vectorstore import create_vectorstore
from src.rag_chain import create_rag_chain

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
UPLOAD_DIR = "data/uploads"

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("📄 Chat with your PDFs")

# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------------------
# Sidebar — Question History
# -------------------------------------------------------------------
st.sidebar.title("📜 Question History")

if st.session_state.history:
    for i, item in enumerate(st.session_state.history, 1):
        with st.sidebar.expander(f"Q{i}: {item['question']}"):
            st.write("**Answer:**")
            st.write(item["answer"])
else:
    st.sidebar.write("No questions asked yet.")

# -------------------------------------------------------------------
# File upload (MULTIPLE PDFs)
# -------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    all_docs = []

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            # Save PDF to disk
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load and split
            try:
                docs = load_and_split_pdf(pdf_path)
                all_docs.extend(docs)

            except ValueError as e:
                st.error(str(e))
                continue

    # Guard: no valid documents
    if not all_docs:
        st.error(
            "No readable text found in the uploaded PDFs.\n\n"
            "Possible reasons:\n"
            "- PDFs are encrypted\n"
            "- PDFs are scanned images\n"
            "- PDFs contain no extractable text"
        )
    else:
        vectorstore = create_vectorstore(all_docs)
        st.session_state.rag_chain = create_rag_chain(vectorstore)
        st.success(f"{len(uploaded_files)} document(s) indexed successfully!")

# -------------------------------------------------------------------
# Question input & answering (PHASE 2: STRING OUTPUT ONLY)
# -------------------------------------------------------------------
if st.session_state.rag_chain:
    st.markdown("---")
    question = st.text_input("Ask a question from the documents")

    if question:
        with st.spinner("Thinking..."):
            result = st.session_state.rag_chain(question)

        answer = result["answer"]
        sources = result["sources"]

        # Save to history (NOW includes sources)
        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })

        st.write("### Answer")
        st.write(answer)

        # 🔎 Evidence section
        st.write("### 📌 Evidence from documents")

        for i, doc in enumerate(sources, 1):
            pdf_path = doc.metadata.get("source")
            page_number = doc.metadata.get("page", 0)
            text_snippet = doc.page_content[:300]  # limit search length

            with st.expander(f"Source {i} — Page {page_number + 1}"):
                try:
                    image = generate_highlight_snapshot(
                        pdf_path=pdf_path,
                        page_number=page_number,
                        highlight_text=text_snippet,
                    )
                    st.image(image, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate snapshot: {e}")
