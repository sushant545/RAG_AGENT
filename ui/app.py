import os
import sys
import time
import streamlit as st

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
from src.loader import load_and_split_pdf
from src.vectorstore import create_vectorstore
from src.rag_chain import create_rag_chain
from src.vision_locator import locate_evidence
from src.pdf_snapshot import render_highlight

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Chat With Your PDF",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Chat With Your PDF")
st.caption(
    "Ask questions • Get answers • Attach visual proof directly to audit reports"
)

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------------------
# Sidebar — History
# -------------------------------------------------------------------
st.sidebar.title("📜 Question History")

if st.session_state.history:
    for i, item in enumerate(st.session_state.history, 1):
        with st.sidebar.expander(f"Q{i}: {item['question']}"):
            st.markdown("**Answer**")
            st.write(item["answer"])

            st.markdown("**Evidence**")
            for ev in item["evidence"]:
                img = render_highlight(
                    ev["pdf_path"],
                    ev["page"],
                    ev["bbox"],
                )
                st.image(img, use_container_width=True)
                st.caption(
                    f"Page {ev['page'] + 1} · "
                    f"Confidence {int(ev['confidence'] * 100)}%"
                )
else:
    st.sidebar.info("No questions asked yet.")

# -------------------------------------------------------------------
# Section 1 — Upload Documents
# -------------------------------------------------------------------
st.markdown("## 📂 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    progress = st.progress(0)
    status = st.empty()

    all_docs = []
    total = len(uploaded_files)

    with st.spinner("Indexing documents…"):
        for idx, file in enumerate(uploaded_files, 1):
            status.info(f"Processing **{file.name}**")
            time.sleep(0.15)  # UX smoothing

            pdf_path = os.path.join(UPLOAD_DIR, file.name)
            with open(pdf_path, "wb") as f:
                f.write(file.read())

            try:
                docs = load_and_split_pdf(pdf_path)
                all_docs.extend(docs)
            except Exception as e:
                st.error(f"{file.name}: {e}")

            progress.progress(idx / total)

        status.success("All documents processed")

    if all_docs:
        st.session_state.vectorstore = create_vectorstore(all_docs)
        st.session_state.rag_chain = create_rag_chain(
            st.session_state.vectorstore
        )
        st.success(f"✅ {len(uploaded_files)} document(s) indexed successfully")
    else:
        st.error("No readable text found in uploaded PDFs")

# -------------------------------------------------------------------
# Section 2 — Search (ALWAYS VISIBLE)
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🔍 Ask Questions?")

search_col, button_col = st.columns([5, 1])

with search_col:
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is my gross salary?",
        label_visibility="collapsed",
    )

with button_col:
    search_clicked = st.button(
        "Search",
        use_container_width=True,
    )

# -------------------------------------------------------------------
# Search execution
# -------------------------------------------------------------------
if search_clicked:
    if not question.strip():
        st.warning("Please enter a question before searching.")
    elif not st.session_state.rag_chain:
        st.warning("Please upload and index documents before searching.")
    else:
        with st.spinner("Analyzing documents and validating evidence…"):
            result = st.session_state.rag_chain(question)

        st.markdown("### ✅ Answer")
        st.write(result["answer"])

        st.markdown("### 📌 Supporting Evidence")

        evidence_items = []

        for doc in result["candidate_docs"]:
            pdf_path = doc.metadata["source"]
            page = doc.metadata["page"]

            try:
                box = locate_evidence(
                    pdf_path,
                    page,
                    question,
                    result["answer"],
                )

                img = render_highlight(
                    pdf_path,
                    page,
                    box,
                )

                st.image(img, use_container_width=True)
                st.caption(
                    f"Page {page + 1} · "
                    f"Confidence {int(box['confidence'] * 100)}%"
                )

                evidence_items.append(
                    {
                        "pdf_path": pdf_path,
                        "page": page,
                        "bbox": box,
                        "confidence": box["confidence"],
                    }
                )

            except Exception as e:
                st.error(f"Failed to generate evidence: {e}")

        # Save to history (audit-safe)
        if evidence_items:
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": result["answer"],
                    "evidence": evidence_items,
                }
            )
