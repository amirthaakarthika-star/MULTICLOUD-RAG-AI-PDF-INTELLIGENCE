import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io
import time

# -------------------- LOAD ENV --------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ Missing GROQ API Key. Please set it in .env file.")
    st.stop()

# -------------------- STREAMLIT SETUP --------------------
st.set_page_config(page_title="MULTI-PDF EXPERT", layout="wide")
st.title("📄 AI POWERED MULTI PDF INTELLIGENCE")

# -------------------- SESSION STATE --------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("🧹 Clear history"):
        st.session_state.chat_history = []

    if st.button("Clear Knowledge Base"):
        try:
            if st.session_state.vector_store is not None:
                st.session_state.vector_store.delete_collection()
            st.session_state.vector_store = None
            st.success("Memory fully cleared!")
        except Exception as e:
            st.error(f"Error clearing DB: {str(e)}")

    if st.button("Build Knowledge Base"):
        if not uploaded_files:
            st.error("Please upload at least one PDF")
        else:
            try:
                if st.session_state.vector_store is not None:
                    st.session_state.vector_store.delete_collection()
                    st.session_state.vector_store = None

                all_docs = []

                with st.spinner("Processing PDFs..."):
                    progress = st.progress(0)
                    total_files = len(uploaded_files)

                    for idx, file in enumerate(uploaded_files):
                        st.write(f"📄 Processing: {file.name}")

                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(file.read())
                                tmp_path = tmp.name

                            loader = PyPDFLoader(tmp_path)
                            docs = loader.load()

                            for d in docs:
                                d.metadata["source"] = file.name

                            all_docs.extend(docs)
                            os.unlink(tmp_path)

                        except Exception as e:
                            st.warning(f"⚠️ Failed: {file.name} → {str(e)}")
                            continue

                        progress.progress((idx + 1) / total_files)

                if not all_docs:
                    st.error("No valid documents processed.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(all_docs)

                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )

                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="pdf_collection",
                    persist_directory="./db"
                )

                st.success(f"Processed {len(uploaded_files)} PDFs successfully!")

            except Exception as e:
                st.error(f"Error building knowledge base: {str(e)}")

    st.markdown("---")
    st.header("💬 Chat History")

    for chat in st.session_state.chat_history:
        with st.expander(f"🧾 {chat['question'][:50]}..."):
            st.markdown("**Full Answer:**")
            st.write(chat['answer'])

# -------------------- MODE SELECTION --------------------
mode = st.radio("Select Mode:", ["General Q&A", "Legal Analysis"])
show_chunks = st.checkbox("🔍 Show Retrieved Context (Debug Mode)")

# -------------------- QUERY --------------------
st.markdown("### 💡 Sample Questions")
st.write("- Summarize the uploaded PDFs")
st.write("- What are the key legal risks?")
st.write("- Compare the uploaded documents")
st.write("- What deadlines are mentioned?")
query = st.text_input("Ask your question:")

if query and st.session_state.vector_store:

    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.3-70b-versatile"
        )

        with st.spinner("Analyzing..."):

            metadatas = st.session_state.vector_store.get()["metadatas"]
            all_sources = list(set([m["source"] for m in metadatas if "source" in m]))

            docs = []
            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6}
            )

            for source in all_sources:
                filtered_docs = retriever.invoke(query, filter={"source": source})
                docs.extend(filtered_docs)

            if not docs:
                st.warning("No relevant information found in documents.")
                st.stop()

            context_by_source = {}
            citations = []
            chunk_debug = []

            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                content = doc.page_content

                context_by_source.setdefault(source, []).append(content)
                citations.append(f"{source} (Page {page})")

                chunk_debug.append({
                    "source": source,
                    "page": page,
                    "content": content[:300]
                })

            structured_context = ""
            for source, texts in context_by_source.items():
                structured_context += f"\n\n===== DOCUMENT: {source} =====\n"
                structured_context += "\n".join(texts[:5])

            # -------------------- PROMPTS --------------------
            if mode == "General Q&A":
                prompt = f"""You are an expert document analyst.
Answer the question strictly based on the provided document context.
Be precise, structured, and cite which document your answer comes from.
If the answer is not found in the documents, clearly state that.

Context:
{structured_context}

Question: {query}

Answer:"""

            else:
                prompt = f"""You are a senior legal analyst with expertise in contract law,
employment law, and corporate compliance.

Analyze the provided legal documents and answer the question with:
1. KEY FINDINGS - Main legal points relevant to the question
2. OBLIGATIONS - What each party is required to do
3. RISKS & RED FLAGS - Potential legal risks or concerning clauses
4. RECOMMENDATIONS - Practical next steps or advice

Always cite which document and section your analysis comes from.
If something is unclear or missing from the documents, explicitly state it.

Legal Documents Context:
{structured_context}

Legal Question: {query}

Legal Analysis:"""

            start = time.time()
            response = llm.invoke(prompt)
            end = time.time()
            answer = response.content

            # -------------------- OUTPUT --------------------
            st.write("### 🧠 Answer")
            st.success("Answer generated successfully")
            st.text_area("Answer", answer, height=250)
            st.caption(f"Response generated in {end - start:.2f} seconds")

            # -------------------- PDF EXPORT --------------------
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = []
            content.append(Paragraph(f"<b>Question:</b> {query}", styles["Normal"]))
            content.append(Paragraph("<br/><br/>", styles["Normal"]))
            content.append(Paragraph(f"<b>Answer:</b> {answer}", styles["Normal"]))

            doc.build(content)

            st.download_button(
                label="⬇️ Download as PDF",
                data=buffer.getvalue(),
                file_name="ai_answer.pdf",
                mime="application/pdf"
            )

            # -------------------- SAVE HISTORY --------------------
            if len(st.session_state.chat_history) == 0 or \
               st.session_state.chat_history[-1]["question"] != query:
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer
                })

            # -------------------- SOURCES --------------------
            st.write("### 📚 Sources & Pages")
            for c in list(set(citations)):
                st.write(f"- {c}")

            # -------------------- DEBUG --------------------
            if show_chunks:
                with st.expander("🔍 Debug Mode (Click to view chunks)"):
                    for i, chunk in enumerate(chunk_debug):
                        st.caption(f"📄 {chunk['source']} (Page {chunk['page']})")
                        st.code(chunk['content'][:300])

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

elif query and not st.session_state.vector_store:
    st.warning("Please upload PDFs and build knowledge base first")