import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
st.title("MULTICLOUD AI POWERED PDF INTELLIGENCE")

#Multi FILE UPLOADER
uploaded_files=st.file_uploader("upload your pdf documents", type="pdf",accept_multiple_files=True)
if uploaded_files:
     all_docs=[]

     with st.spinner("AI is analyzing all documents..."):
         for uploaded_file in uploaded_files:
            #save each file temporary with manual close to avoid file lock errors
             with tempfile.NamedTemporaryFile(delete=False)as tmp_file:
              tmp_file.write(uploaded_file.read())
              path=tmp_file.name
              tmp_file.close()
         #load and extract file
             loader=PyPDFLoader(path)
             data=loader.load()
             all_docs.extend(data)#end of the loop
         #clean up temporary file
             os.remove(path)
         splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
         docs=splitter.split_documents(all_docs)
                   
         embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")         
         vectorstore=Chroma.from_documents(documents=docs,embedding=embeddings)
#setup brainpower
         llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
         qa_chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=vectorstore.as_retriever(search_kwargs={"k":10}))
         st.success(f"Ready!Analyzed{len(uploaded_files)} document(s).")
#add a clar button in sidebar or mainpage
if st.button("clear chat"):
    st.rerun()#this refreshes the page and clear all the ios
#chat box logic
user_q=st.text_input("Ask a question about this pdfs:")
if user_q:
    #using .invoke for modern langchain standards
    result=qa_chain.invoke({"query": user_q})
    #safeguard the dictinary access
    st.info(result.get("result","Answer not found in context."))


                         



                

   


