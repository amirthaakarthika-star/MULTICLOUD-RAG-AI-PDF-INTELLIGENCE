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
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
st.set_page_config(page_title="MULTI-PDF EXPERT",layout="wide")
st.title("AI POWERED MULTI PDF INTELLIGENCE")
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None
with st.sidebar:
    st.header("upload documents")
    uploaded_files=st.file_uploader("select multiple pdfs", type="pdf",accept_multiple_files=True)
    if st.button("Build Knowledge Base"):
        if uploaded_files:
            all_documents=[]
            with st.spinner("merging PDFS into one brain..."):
                for uploaded_file in uploaded_files:   
                    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
                         tmp.write(uploaded_file.read())
                    loader=PyPDFLoader(tmp.name)
                    all_documents.extend(loader.load())
                    os.unlink(tmp.name)
                splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
                chunks=splitter.split_documents(all_documents)
                embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_store=Chroma.from_documents(chunks,embeddings)
                st.success(f"Ready! combined{len(uploaded_files)} files into{len(chunks)} chunks.")
        else:
         st.error("upload files first!")
query=st.text_input("Ask a question from this pdfs:")
if query and st.session_state.vector_store:
             retriever = st.session_state.vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 20,"fetch_k": 50, "lambda_mult": 0.5}) 
             llm=ChatGroq(groq_api_key=groq_api_key,model="llama-3.3-70b-versatile")
             qa=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)
             with st.spinner("searching accross all files..."):
                  result=qa.invoke({"query":query})   
                  st.write("###AI Analysis:")
                  st.info(result["result"])   