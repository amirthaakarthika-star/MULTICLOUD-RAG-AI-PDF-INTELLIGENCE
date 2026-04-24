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
    st.error("Missing GROQ API Key. Please set it in .env file.")
    st.stop()

# ---------- STREAMLIT SETUP ----------
st.set_page_config(page_title="MULTI-PDF EXPERT", layout="wide")

# ---------- STYLING ----------
st.markdown("""
<style>
    .hero-box {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(34, 211, 238, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin-bottom: 2rem;
        color: #F4F4F4;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #22D3EE, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 15px;
    }
    .badge {
        padding: 5px 15px;
        border-radius: 12px;
        background: rgba(34, 211, 238, 0.1);
        border: 1px solid rgba(34, 211, 238, 0.5);
        font-size: 0.85rem;
        font-weight: 600;
        color: #22D3EE;
    }
</style>

<div class="hero-box">
    <div class="hero-title">Multi-PDF Intelligence Hub</div>
    <div style="font-size: 1.1rem; opacity: 0.8;">
        Next-gen RAG Analysis with Groq-speed inference and Legal-grade precision.
    </div>
    <div class="badge-row">
        <div class="badge">Groq LPU</div>
        <div class="badge">LangChain</div>
        <div class="badge">Multi-PDF</div>
        <div class="badge">Legal Mode</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query" not in st.session_state:
    st.session_state.query = ""

if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""

if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

if "latest_chunk_debug" not in st.session_state:
    st.session_state.latest_chunk_debug = []

if "latest_elapsed" not in st.session_state:
    st.session_state.latest_elapsed = None

if "last_processed_query" not in st.session_state:
    st.session_state.last_processed_query = ""

if "download_pdf_bytes" not in st.session_state:
    st.session_state.download_pdf_bytes = None


def set_sample_query(sample_query: str) -> None:
    st.session_state.query = sample_query


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Clear history"):
        st.session_state.chat_history = []

    if st.button("Clear Knowledge Base"):
        try:
            if st.session_state.vector_store is not None:
                st.session_state.vector_store.delete_collection()
            st.session_state.vector_store = None
            st.session_state.latest_answer = ""
            st.session_state.latest_sources = []
            st.session_state.latest_chunk_debug = []
            st.session_state.latest_elapsed = None
            st.session_state.download_pdf_bytes = None
            st.session_state.last_processed_query = ""
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
                        st.write(f"Processing: {file.name}")

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
                            st.warning(f"Failed: {file.name} -> {str(e)}")
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
    st.header("Chat History")

    for chat in st.session_state.chat_history:
        with st.expander(f"{chat['question'][:50]}..."):
            st.markdown("**Full Answer:**")
            st.write(chat["answer"])

# -------------------- MODE SELECTION --------------------
mode = st.radio("Select Mode:", ["General Q&A", "Legal Analysis"])
show_chunks = st.checkbox("Show Retrieved Context (Debug Mode)")

# ---------- QUERY & INTERACTIVE SAMPLES ----------
st.markdown("### Quick Analysis")

with st.container():
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.button(
            "Summarize Documents",
            use_container_width=True,
            on_click=set_sample_query,
            args=("Summarize the uploaded PDFs in detail.",)
        )
    with col2:
        st.button(
            "Legal Risk Analysis",
            use_container_width=True,
            on_click=set_sample_query,
            args=("What are the key legal risks found in these documents?",)
        )
    with col3:
        st.button(
            "Compare Documents",
            use_container_width=True,
            on_click=set_sample_query,
            args=("Compare the uploaded documents and highlight the main differences.",)
        )
    with col4:
        st.button(
            "Check Deadlines",
            use_container_width=True,
            on_click=set_sample_query,
            args=("What are the important dates and deadlines mentioned in these files?",)
        )

query = st.text_input("Ask your question:", key="query")
run_analysis = st.button("Analyze Question", type="primary")

if run_analysis:
    if not query:
        st.warning("Please enter a question first")
    elif not st.session_state.vector_store:
        st.warning("Please upload PDFs and build knowledge base first")
    else:
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

                for retrieved_doc in docs:
                    source = retrieved_doc.metadata.get("source", "Unknown")
                    page = retrieved_doc.metadata.get("page", "N/A")
                    content = retrieved_doc.page_content

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

                if mode == "General Q&A":
                    prompt = f"Answer from context:\n{structured_context}\n\nQuestion: {query}"
                else:
                    prompt = f"Legal analysis:\n{structured_context}\n\nQuestion: {query}"

                start = time.time()
                response = llm.invoke(prompt)
                end = time.time()
                answer = response.content

                buffer = io.BytesIO()
                pdf_doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()

                pdf_content = [
                    Paragraph(f"<b>Question:</b> {query}", styles["Normal"]),
                    Paragraph("<br/><br/>", styles["Normal"]),
                    Paragraph(f"<b>Answer:</b> {answer}", styles["Normal"])
                ]
                pdf_doc.build(pdf_content)

                st.session_state.latest_answer = answer
                st.session_state.latest_sources = list(set(citations))
                st.session_state.latest_chunk_debug = chunk_debug
                st.session_state.latest_elapsed = end - start
                st.session_state.download_pdf_bytes = buffer.getvalue()
                st.session_state.last_processed_query = query

                if len(st.session_state.chat_history) == 0 or st.session_state.chat_history[-1]["question"] != query:
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer
                    })

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# -------------------- OUTPUT --------------------
if st.session_state.latest_answer:
    st.write("### Answer")
    st.success("Answer generated successfully")
    st.text_area("Answer", st.session_state.latest_answer, height=250)

    if st.session_state.latest_elapsed is not None:
        st.caption(f"Response generated in {st.session_state.latest_elapsed:.2f} seconds")

    if st.session_state.download_pdf_bytes is not None:
        st.download_button(
            label="Download as PDF",
            data=st.session_state.download_pdf_bytes,
            file_name="ai_answer.pdf",
            mime="application/pdf"
        )

    st.write("### Sources & Pages")
    for citation in st.session_state.latest_sources:
        st.write(f"- {citation}")

    if show_chunks:
        with st.expander("Debug Mode (Click to view chunks)"):
            for chunk in st.session_state.latest_chunk_debug:
                st.caption(f"{chunk['source']} (Page {chunk['page']})")
                st.code(chunk["content"])
