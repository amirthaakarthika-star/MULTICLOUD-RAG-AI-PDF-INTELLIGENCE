import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit setup
st.set_page_config(page_title="MULTI-PDF EXPERT", layout="wide")
st.title("📄 AI POWERED MULTI PDF INTELLIGENCE")

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar
with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    # ✅ CLEAR MEMORY (FIXED)
    if st.button("Clear Knowledge Base"):
        if st.session_state.vector_store is not None:
            st.session_state.vector_store.delete_collection()  # 🔥 FULL RESET

        st.session_state.vector_store = None
        st.success("Memory fully cleared!")

    # Build Knowledge Base
    if st.button("Build Knowledge Base"):
        if uploaded_files:

            # 🔥 SAFETY RESET BEFORE NEW BUILD (VERY IMPORTANT)
            if st.session_state.vector_store is not None:
                st.session_state.vector_store.delete_collection()
                st.session_state.vector_store = None

            all_docs = []

            with st.spinner("Processing PDFs..."):
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    for d in docs:
                        d.metadata["source"] = file.name

                    all_docs.extend(docs)
                    os.unlink(tmp_path)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(all_docs)

                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )

                # 🔥 NEW CLEAN VECTOR STORE (NO OLD DATA MIXING)
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="pdf_collection"
                )

                st.success(f"Processed {len(uploaded_files)} PDFs successfully!")

        else:
            st.error("Please upload at least one PDF")

# Query section
query = st.text_input("Ask your question:")

if query and st.session_state.vector_store:

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.3-70b-versatile"
    )

    with st.spinner("Analyzing..."):

        # 🔥 GET ALL SOURCES FROM VECTOR DB
        metadatas = st.session_state.vector_store.get()["metadatas"]
        all_sources = list(set([m["source"] for m in metadatas if "source" in m]))

        docs = []

        # 🔥 FORCE RETRIEVAL PER DOCUMENT (CRITICAL FIX)
        for source in all_sources:
            filtered_docs = st.session_state.vector_store.similarity_search(
                query,
                k=4,
                filter={"source": source}
            )
            docs.extend(filtered_docs)

        # Group by source
        context_by_source = {}
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            context_by_source.setdefault(source, []).append(doc.page_content)

        # Build context
        structured_context = ""
        for source, texts in context_by_source.items():
            structured_context += f"\n\n===== DOCUMENT: {source} =====\n"
            structured_context += "\n".join(texts[:5])

        # MODE SWITCH
        if len(context_by_source) == 1:
            st.caption("🟢 Mode: Single PDF Q&A")

            prompt = f"""
You are an AI assistant answering from a single PDF.

{structured_context}

Question: {query}

Instructions:
- Answer clearly and accurately
- Use only the document
"""

        else:
            st.caption("🔵 Mode: Multi-PDF Comparison")

            prompt = f"""
You are an AI that compares multiple PDF documents.

{structured_context}

Question: {query}

Instructions:
- Compare ALL documents properly
- Mention each document name
- Highlight similarities and differences
- If missing info, say "Not found in [document]"
"""

        response = llm.invoke(prompt)

        st.write("### 🧠 Answer")
        st.info(response.content)

        st.write("### 📚 Documents Used")
        for s in context_by_source.keys():
            st.write(f"- {s}")

elif query and not st.session_state.vector_store:
    st.warning("Please upload PDFs and build knowledge base first")