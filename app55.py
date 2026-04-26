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

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Powered Multi PDF Intelligence",
    page_icon="📄",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
# ---------- CLEAN PROFESSIONAL HEADER ----------
st.title("📄 AI Powered Multi PDF Intelligence")

st.markdown(
    "Analyze multiple PDFs using AI-powered search, comparison, legal review, and fast chat responses."
)

st.write("")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.success("⚡ Groq Powered")

with col2:
    st.info("🧠 RAG Search")

with col3:
    st.info("📚 Multi PDF")

with col4:
    st.warning("⚖️ Legal Mode")

with col5:
    st.success("🚀 Streamlit")

st.divider()

# -------------------- CHECK API KEY --------------------
if not groq_api_key:
    st.error("❌ Missing GROQ API Key. Add it in .env or Streamlit secrets.")
    st.stop()

# -------------------- SESSION STATE --------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []

    if st.button("🗑️ Clear Knowledge Base"):
        try:
            if st.session_state.vector_store:
                st.session_state.vector_store.delete_collection()
            st.session_state.vector_store = None
            st.success("Knowledge base cleared.")
        except:
            pass

    if st.button("🚀 Build Knowledge Base"):
        if not uploaded_files:
            st.error("Please upload PDFs first.")
        else:
            try:
                all_docs = []

                with st.spinner("Processing PDFs..."):
                    progress = st.progress(0)
                    total = len(uploaded_files)

                    for i, file in enumerate(uploaded_files):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()

                        for d in docs:
                            d.metadata["source"] = file.name

                        all_docs.extend(docs)
                        os.unlink(tmp_path)

                        progress.progress((i + 1) / total)

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

                st.success(f"✅ {len(uploaded_files)} PDFs processed successfully!")

            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.header("🕘 Chat History")

    for chat in st.session_state.chat_history:
        with st.expander(chat["question"][:50]):
            st.write(chat["answer"])

# -------------------- STATUS --------------------
if not st.session_state.vector_store:
    st.info("⬅️ Upload PDFs and click **Build Knowledge Base** to begin.")
else:
    st.success("✅ Knowledge Base Ready")

# -------------------- MAIN CONTROLS --------------------
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("💬 Ask a question about your PDFs")

with col2:
    mode = st.selectbox("Mode", ["General Q&A", "Legal Analysis"])

show_chunks = st.checkbox("🔍 Show Debug Context")

# -------------------- QUICK ACTION BUTTONS --------------------
st.markdown("### ⚡ Quick Prompts")

c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("Summarize"):
        query = "Summarize the uploaded PDFs"

with c2:
    if st.button("Compare"):
        query = "Compare the uploaded documents"

with c3:
    if st.button("Risks"):
        query = "What are the key risks?"

with c4:
    if st.button("Deadlines"):
        query = "What deadlines are mentioned?"

# -------------------- QUERY EXECUTION --------------------
if query and st.session_state.vector_store:

    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.3-70b-versatile"
        )

        with st.spinner("Analyzing documents..."):

            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 10,
                    "fetch_k": 25,
                    "lambda_mult": 0.7
                }
            )

            docs = retriever.invoke(query)

            if not docs:
                st.warning("No relevant content found.")
                st.stop()

            citations = []
            chunk_debug = []
            context = ""

            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")

                context += f"\n[{source} - Page {page}]\n{doc.page_content}\n"

                citations.append(f"{source} (Page {page})")

                chunk_debug.append({
                    "source": source,
                    "page": page,
                    "content": doc.page_content[:300]
                })

            # Smart compare prompt
            if "compare" in query.lower():
                prompt = f"""
Compare information from ALL uploaded PDFs using only the context below.

Mention each source file separately.
Highlight similarities, differences, risks, deadlines, clauses, or important points.
If one PDF lacks information, state that clearly.

Context:
{context}

Question: {query}
"""
            elif mode == "General Q&A":
                prompt = f"""
Answer only from the context below.

Context:
{context}

Question: {query}
"""
            else:
                prompt = f"""
Provide legal analysis using only the context below.

Context:
{context}

Question: {query}
"""

            start = time.time()
            response = llm.invoke(prompt)
            end = time.time()

            answer = response.content
            # ---------------- OUTPUT ----------------
            st.markdown("## 🧠 Answer")
            st.success("Answer generated successfully")

            st.text_area("Response", answer, height=260)

            st.caption(f"⚡ Generated in {end-start:.2f} sec")

            # ---------------- PDF EXPORT ----------------
            buffer = io.BytesIO()
            pdf = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = [
                Paragraph(f"<b>Question:</b> {query}", styles["Normal"]),
                Paragraph("<br/><br/>", styles["Normal"]),
                Paragraph(f"<b>Answer:</b> {answer}", styles["Normal"])
            ]

            pdf.build(content)

            st.download_button(
                "⬇️ Download Answer as PDF",
                data=buffer.getvalue(),
                file_name="ai_answer.pdf",
                mime="application/pdf"
            )

            # ---------------- SAVE HISTORY ----------------
            if len(st.session_state.chat_history) == 0 or \
               st.session_state.chat_history[-1]["question"] != query:

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer
                })

            # ---------------- SOURCES ----------------
            st.markdown("### 📚 Sources")

            for c in list(set(citations)):
                st.write("- " + c)

            # ---------------- DEBUG ----------------
            if show_chunks:
                with st.expander("🔍 Retrieved Chunks"):
                    for chunk in chunk_debug:
                        st.caption(f"{chunk['source']} | Page {chunk['page']}")
                        st.code(chunk["content"])

    except Exception as e:
        st.error(str(e))

elif query and not st.session_state.vector_store:
    st.warning("Please upload PDFs and build knowledge base first.")