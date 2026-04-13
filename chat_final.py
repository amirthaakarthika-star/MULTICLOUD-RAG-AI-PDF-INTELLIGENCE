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
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    # Clear memory
    if st.button("🧹 Clear Knowledge Base"):
        st.session_state.vector_store = None
        st.success("Memory cleared!")

    # Build knowledge base
    if st.button("⚙️ Build Knowledge Base"):
        if uploaded_files:
            all_documents = []

            with st.spinner("Processing PDFs..."):
                for uploaded_file in uploaded_files:
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    # Add source metadata
                    for d in docs:
                        d.metadata["source"] = uploaded_file.name

                    all_documents.extend(docs)
                    os.unlink(tmp_path)

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(all_documents)

                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )

                # 🔥 Reset vector store (fix memory issue)
                st.session_state.vector_store = Chroma.from_documents(
                    chunks,
                    embeddings
                )

                st.success(f"✅ {len(uploaded_files)} PDF(s) processed into {len(chunks)} chunks.")
        else:
            st.error("Please upload at least one PDF.")

# Main query
query = st.text_input("💬 Ask a question across your PDFs:")

if query and st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": 10}
    )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.3-70b-versatile"
    )

    with st.spinner("Analyzing documents..."):
        docs = retriever.invoke(query)

        # Group by source
        context_by_source = {}
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in context_by_source:
                context_by_source[source] = []
            context_by_source[source].append(doc.page_content)

        # Build structured context
        structured_context = ""
        for source, texts in context_by_source.items():
            structured_context += f"\n\n===== DOCUMENT: {source} =====\n"
            structured_context += "\n".join(texts[:5])

        # 🔥 MODE SWITCH (single vs multi PDF)
        if len(context_by_source) == 1:
            st.caption("🟢 Mode: Single Document Q&A")

            prompt = f"""
You are an AI assistant answering questions from a single PDF.

{structured_context}

Question: {query}

Instructions:
- Answer clearly and accurately
- Use only the provided document
- Be concise and helpful
"""
        else:
            st.caption("🔵 Mode: Multi-Document Comparison")

            prompt = f"""
You are an AI that compares multiple PDF documents.

{structured_context}

Question: {query}

Instructions:
- Compare information across documents
- Clearly mention document names
- Highlight similarities and differences
- If info is missing in a document, say:
  "Not found in [document name]"
- Structure the answer neatly
"""

        # LLM response
        response = llm.invoke(prompt)

        # Output
        st.write("### 🧠 AI Analysis")
        st.info(response.content)

        st.write("### 📚 Documents Used")
        for source in context_by_source.keys():
            st.write(f"- {source}")

elif query and not st.session_state.vector_store:
    st.warning("⚠️ Please upload and process PDFs first.")