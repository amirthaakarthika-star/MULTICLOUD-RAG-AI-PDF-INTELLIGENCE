import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="MULTI-PDF EXPERT", layout="wide")
st.title("AI POWERED MULTI PDF INTELLIGENCE")

# Initialize vector store in session
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader("Select multiple PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Build Knowledge Base"):
        if uploaded_files:
            all_documents = []
            with st.spinner("Merging PDFs into one knowledge base..."):
                for uploaded_file in uploaded_files:
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    # Add metadata for source tracking
                    for d in docs:
                        d.metadata["source"] = uploaded_file.name

                    all_documents.extend(docs)
                    os.unlink(tmp_path)

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(all_documents)

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                # Build or extend vector store
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = Chroma.from_documents(chunks, embeddings)
                else:
                    st.session_state.vector_store.add_documents(chunks)

                st.success(f"Ready! Combined {len(uploaded_files)} files into {len(chunks)} chunks.")
        else:
            st.error("Upload files first!")

# Query interface
query = st.text_input("Ask a question across all PDFs:")
if query and st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 50, "lambda_mult": 0.5}
    )
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    with st.spinner("Searching across all files..."):
        result = qa.invoke({"query": query})

        st.write("### AI Analysis:")
        st.info(result["result"])

        st.write("### Sources:")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata['source']}")