import streamlit as st
import os
import shutil
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain.schema import BaseRetriever
from typing import List
from langchain.docstore.document import Document

st.title("MULTICLOUD AI POWERED PDF INTELLIGENCE")import streamlit as st


st.title("MULTICLOUD AI POWERED PDF INTELLIGENCE")

# Session state
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
if "sources" not in st.session_state: st.session_state.sources = set()

uploaded_files = st.file_uploader("Upload your PDF documents (test with 2+)", type="pdf", accept_multiple_files=True)

# Analyze button
if st.button("🔄 Analyze All PDFs") and uploaded_files:
    all_docs = []
    filenames = [f.name for f in uploaded_files]  # Track names
    st.session_state.sources = set(filenames)
    
    with st.spinner(f"Processing {len(uploaded_files)} PDFs..."):
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filenames[i]}") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name
            
            loader = PyPDFLoader(path)
            docs = loader.load()
            # Ensure source metadata (PyPDFLoader sets it to path)
            for doc in docs:
                doc.metadata['source'] = filenames[i]  # Override to filename for clarity
            all_docs.extend(docs)
            os.unlink(path)
    
    # Smaller chunks for better granularity
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.split_documents(all_docs)
    st.write(f"📊 Created {len(docs)} chunks from {len(filenames)} files")
    
    # Embed & store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir): shutil.rmtree(persist_dir)
    st.session_state.vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    
    # Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0)
    
    # Custom retriever for multi-source enforcement
    def multi_source_retriever(query: str, k: int = 30) -> List[Document]:
        docs = st.session_state.vectorstore.similarity_search(query, k=k)
        # Group by source, take top 5 per source min
        source_docs = {}
        for doc in docs:
            src = doc.metadata.get('source', 'unknown')
            if src not in source_docs:
                source_docs[src] = []
            source_docs[src].append(doc)
        
        balanced_docs = []
        for src, sdocs in source_docs.items():
            balanced_docs.extend(sdocs[:6])  # 6 per source
            if len(balanced_docs) >= 25: break
        return balanced_docs[:25]  # Cap context
    
    # Custom retriever wrapper
    class MultiSourceRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> List[Document]:
            return multi_source_retriever(query)
    
    retriever = MultiSourceRetriever()
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    st.success("✅ Multi-PDF ready! Sources: " + ", ".join(sorted(st.session_state.sources)))

# Clear
if st.button("🗑️ Clear All"): 
    if os.path.exists("./chroma_db"): shutil.rmtree("./chroma_db")
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

# Status
if st.session_state.vectorstore:
    st.info(f"Active sources: {list(st.session_state.sources)}")

# Query
user_q = st.text_input("Ask spanning multiple PDFs:")
if user_q and st.session_state.qa_chain:
    with st.spinner("Retrieving from ALL PDFs..."):
        result = st.session_state.qa_chain.invoke({"query": user_q})
        st.markdown("**Answer:**")
        st.info(result.get("result", "No context found."))
    
    # Debug: Show retrieved sources
    docs = multi_source_retriever(user_q)
    with st.expander("🔍 Retrieved Sources (confirms multi-PDF)"):
        sources = {}
        for doc in docs:
            src = doc.metadata.get('source', '?')
            sources[src] = sources.get(src, 0) + 1
        for src, count in sources.items():
            st.write(f"• **{src}**: {count} chunks")
        st.caption(f"Total chunks: {len(docs)}")

elif user_q:
    st.error("Analyze PDFs first!")

# Session state
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
if "sources" not in st.session_state: st.session_state.sources = set()

uploaded_files = st.file_uploader("Upload your PDF documents (test with 2+)", type="pdf", accept_multiple_files=True)

# Analyze button
if st.button("🔄 Analyze All PDFs") and uploaded_files:
    all_docs = []
    filenames = [f.name for f in uploaded_files]  # Track names
    st.session_state.sources = set(filenames)
    
    with st.spinner(f"Processing {len(uploaded_files)} PDFs..."):
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filenames[i]}") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name
            
            loader = PyPDFLoader(path)
            docs = loader.load()
            # Ensure source metadata (PyPDFLoader sets it to path)
            for doc in docs:
                doc.metadata['source'] = filenames[i]  # Override to filename for clarity
            all_docs.extend(docs)
            os.unlink(path)
    
    # Smaller chunks for better granularity
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.split_documents(all_docs)
    st.write(f"📊 Created {len(docs)} chunks from {len(filenames)} files")
    
    # Embed & store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir): shutil.rmtree(persist_dir)
    st.session_state.vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    
    # Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0)
    
    # Custom retriever for multi-source enforcement
    def multi_source_retriever(query: str, k: int = 30) -> List[Document]:
        docs = st.session_state.vectorstore.similarity_search(query, k=k)
        # Group by source, take top 5 per source min
        source_docs = {}
        for doc in docs:
            src = doc.metadata.get('source', 'unknown')
            if src not in source_docs:
                source_docs[src] = []
            source_docs[src].append(doc)
        
        balanced_docs = []
        for src, sdocs in source_docs.items():
            balanced_docs.extend(sdocs[:6])  # 6 per source
            if len(balanced_docs) >= 25: break
        return balanced_docs[:25]  # Cap context
    
    # Custom retriever wrapper
    class MultiSourceRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> List[Document]:
            return multi_source_retriever(query)
    
    retriever = MultiSourceRetriever()
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    st.success("✅ Multi-PDF ready! Sources: " + ", ".join(sorted(st.session_state.sources)))

# Clear
if st.button("🗑️ Clear All"): 
    if os.path.exists("./chroma_db"): shutil.rmtree("./chroma_db")
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

# Status
if st.session_state.vectorstore:
    st.info(f"Active sources: {list(st.session_state.sources)}")

# Query
user_q = st.text_input("Ask spanning multiple PDFs:")
if user_q and st.session_state.qa_chain:
    with st.spinner("Retrieving from ALL PDFs..."):
        result = st.session_state.qa_chain.invoke({"query": user_q})
        st.markdown("**Answer:**")
        st.info(result.get("result", "No context found."))
    
    # Debug: Show retrieved sources
    docs = multi_source_retriever(user_q)
    with st.expander("🔍 Retrieved Sources (confirms multi-PDF)"):
        sources = {}
        for doc in docs:
            src = doc.metadata.get('source', '?')
            sources[src] = sources.get(src, 0) + 1
        for src, count in sources.items():
            st.write(f"• **{src}**: {count} chunks")
        st.caption(f"Total chunks: {len(docs)}")

elif user_q:
    st.error("Analyze PDFs first!")