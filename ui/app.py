import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st

from src.loader import load_and_split_pdf
from src.vectorstore import create_vectorstore
from src.rag_chain import create_rag_chain

UPLOAD_DIR = "data/uploads"

st.set_page_config(page_title="PDF RAG Chat", layout="centered")
st.title("📄 Chat with your PDF")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        docs = load_and_split_pdf(pdf_path)
        vectorstore = create_vectorstore(docs)
        st.session_state.rag_chain = create_rag_chain(vectorstore)

    st.success("Document indexed! Ask questions below.")

if st.session_state.rag_chain:
    question = st.text_input("Ask a question from the document")

    if question:
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_chain.invoke(question)
        st.write("### Answer")
        st.write(answer)
