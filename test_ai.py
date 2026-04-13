import streamlit as st
import os
import shutil  # For safe cleanup
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA  # Modern import

st.title("MULTICLOUD AI POWERED PDF INTELLIGENCE")

# Session state for persistence
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "num_docs" not in st.session_state:
    st.session_state.num_docs = 0

# Multi-file uploader
uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

# Process button - forces re-analysis even with existing store
if st.button("🔄 Analyze/Re-analyze All PDFs") and uploaded_files:
    if uploaded_files:
        all_docs = []
        with st.spinner(f"AI is analyzing {len(uploaded_files)} documents..."):
            for uploaded_file in uploaded_files:
                # Save to temp file (PyPDFLoader needs file path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    path = tmp_file.name

                # Load and extract text/pages
                loader = PyPDFLoader(path)
                data = loader.load()
                all_docs.extend(data)
                
                # Safe cleanup right after
                os.unlink(path)

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(all_docs)
            
            # Embeddings and vectorstore (rebuilds fully for all docs)
            embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
            
            persist_dir = "./chroma_db"
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
            st.session_state.vectorstore = Chroma.from_documents(
                documents=docs, embedding=embeddings, persist_directory=persist_dir
            )
            
            # Setup QA chain with broader retrieval
            llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0)
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 20})
            )
            st.session_state.num_docs = len(uploaded_files)
            st.success(f"✅ Ready! Indexed {len(docs)} chunks from {len(uploaded_files)} PDFs.")

# Clear everything
if st.button("🗑️ Clear Chat & Data"):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.num_docs = 0
    st.rerun()

# Status display
if st.session_state.vectorstore:
    st.info(f"📚 Active: {st.session_state.num_docs} PDFs indexed. Ask away!")
else:
    st.warning("👆 Upload PDFs and click 'Analyze' first.")

# Chat input
user_q = st.text_input("Ask a question about these PDFs:")
if user_q and st.session_state.qa_chain:
    with st.spinner("Generating answer with all PDF contexts..."):
        result = st.session_state.qa_chain.invoke({"query": user_q})
        st.info(result.get("result", "Answer not found in context."))
        
        # Show sources for verification (helps debug multi-PDF)
        try:
            docs = st.session_state.qa_chain.retriever.get_relevant_documents(user_q)
            with st.expander("📄 Sources (top 3 chunks)"):
                for i, doc in enumerate(docs[:3]):
                    source = doc.metadata.get("source", "Unknown")
                    st.write(f"**Chunk {i+1}:** {source[:100]}...")
                    st.caption(doc.page_content[:200] + "...")
        except:
            st.caption("Sources unavailable.")
elif user_q:
    st.warning("Please analyze PDFs first.")