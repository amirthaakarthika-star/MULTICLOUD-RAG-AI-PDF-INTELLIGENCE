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
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI

if "vector_store"not in st.session_state:
    st.session_state.vector_store=None
st.title("MULTICLOUD PDF ANALYZER")
uploaded_files=st.file_uploader("upload your pdf documents", type="pdf", accept_multiple_files=True)

if st.button("Analyze All PDFs"):
    if uploaded_files:
        all_docs=[]
        with st.spinner(f"AI is analyzing{len(uploaded_files)}documents..."):

        for uploaded_file in uploaded_files:
                # Save to temp file (PyPDFLoader needs file path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    path = tmp_file.name
                     
                loader=PyPDFLoader(path)
                all_docs.extend(loader.load())
                os.unlink(path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_docs)
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
        st.session_state.vector_store=chroma.from_documents(documents=docs,embedding=embeddings)
        st.success("ANALYSES COMPLETE YOU CAN NOW ASK QUESTIONS ABOUT ALL FILES.")
query=st.text_input("Ask a questions about your documents:")
if query and st.session_state.vector_store:
    retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k":10})
    llm=ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)

    response = qa_chain.invoke(query)
    st.write(response["result"])
elif query:
    st.warning("please upload and analyze documents first.")    

                         
        