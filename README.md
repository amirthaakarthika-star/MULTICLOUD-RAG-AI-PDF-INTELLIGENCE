# 📄 RETRIEVAL AUGMENTED MULTI PDF INTELLIGENCE SYSTEM

## 🎯 Project Overview

This project is a Retrieval-Augmented Generation (RAG) based PDF Analysis Chatbot designed to help users extract insights from one or multiple PDF documents using natural language queries.

Users can upload PDFs, ask context-aware questions, and receive accurate responses with source citations. The system improves answer reliability by retrieving relevant document chunks before generating responses.

### Key Technical Highlights

- **Multi-PDF Support:** Upload and analyze multiple PDF files simultaneously.
- **Semantic Search:** Uses embeddings and vector similarity search to retrieve relevant content.
- **Reduced Hallucinations:** Grounds responses using retrieved document context.
- **Interactive UI:** Built with Streamlit for a smooth user experience.
- **Explainability:** Displays source citations and optional debug retrieval view.
- **Export Support:** Users can export answers as PDF.

---
![App Interface](cleanui.png)

---

## 🔥 Features
- 📄 Multi-PDF upload and analysis
- ⚖️ Legal Analysis Mode (obligations, risks, clauses)
- 🔍 Multi-document comparison
- 📚 Page-level citations
- 🧠 Explainable AI (debug view showing retrieved chunks)
- ⚡ MMR-based retrieval for better relevance
- 🛠️ Error handling and robust processing

- Retrieval-Augmented Generation (RAG)
- Legal document analysis mode
- Debug mode with source tracing to inspect retrieved chunks

- Export answers as PDF

- 🤖 Ask questions across all PDF
- 🧠 Chat history in sidebar(session based)
  
- 🔍 Source citation with page numbers
- ⚡ Fast semantic search using embeddings
- RAG-based accurate answers with citations
- Response time display
- Clean Streamlit UI
   
---

## 🧠 Tech Stack
| Category | Tool / Technology | Professional Purpose |
| :--- | :--- | :--- |
| **Language** | Python | Primary backend development and data processing. |
| **Frontend** | Streamlit | Architecting a responsive, real-time user interface. |
| **Orchestration** | LangChain | Managing complex RAG workflows and document chains. |
| **Vector Database** | ChromaDB | High-performance semantic storage and retrieval. |
| **Embeddings** | HuggingFace | Generating high-dimensional vectors for text intelligence. |
| **Inference** | Groq LLM | Leveraging ultra-low latency hardware for rapid AI responses. |
| **Processing** | PyPDF | Handling unstructured data extraction from PDF sources. |
| **Output** | ReportLab | Programmatic generation of PDF reports from AI insights. |
| **DevOps** | GitHub | Version control and project lifecycle management. |


## 🏗️ System Architecture
flowchart TD
    subgraph Client_Interface [User Layer]
    A[User Uploads PDFs]
    F[User Query]
    end

    subgraph Ingestion_Pipeline [Data Processing]
    B[PDF Loader] --> C[Recursive Text Splitter]
    C --> D[Embeddings Model]
    end

    subgraph Storage [Vector Database]
    D --> E[(ChromaDB)]
    end

    subgraph Inference_Engine [AI Core]
    F --> G[Retriever: MMR Strategy]
    E --> G
    G --> H[Groq Cloud: Llama 3]
    H --> I[Grounded Answer]
    end

    subgraph Export_Layer [Output]
    I --> J[Chat History / PDF Export]
    end

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#dfd,stroke:#333,stroke-width:2px
    




---
## 🧠 Engineering Challenges & Solutions

### 1. Retrieval Quality Across Multiple PDFs

**The Challenge:**
When querying multiple uploaded PDFs, early retrieval results sometimes favored chunks from one dominant document, causing incomplete answers or missed information from other files.

**The Solution:**

* Tuned `chunk_size` and `chunk_overlap` for better context preservation.
* Optimized retriever search parameters through iterative testing.
* Replaced basic similarity search with **Maximal Marginal Relevance (MMR)** retrieval.
* MMR improves diversity by selecting relevant chunks while reducing repetitive results from the same source.

**Outcome:**
More balanced retrieval across multiple PDFs and stronger cross-document question answering.

---

### 2. API Quota Limits & Response Speed

**The Challenge:**
Initial development using Google Gemini encountered free-tier quota limits and interrupted testing workflows.

**The Solution:**

* Evaluated alternative LLM providers based on speed and usability.
* Migrated inference to **Groq API** for faster response times.
* Moved API credentials to environment variables for easier configuration and provider switching.

**Outcome:**
Smoother development cycles, faster responses, and improved user experience.

---

### 3. Managing Rapidly Changing Dependencies

**The Challenge:**
Frequent LangChain updates, renamed modules, and dependency changes created import and compatibility issues during development.

**The Solution:**

* Updated imports to match the latest package structure.
* Pinned stable versions where needed.
* Improved maintainability through modular configuration.

**Outcome:**
Better project stability and easier future upgrades.

---

### 4. Performance Optimization for Large PDFs

**The Challenge:**
Processing large PDF files and multiple uploads increased memory usage and slowed response workflows.

**The Solution:**

* Optimized document ingestion flow.
* Reduced unnecessary overhead in retrieval steps.
* Used efficient vector search workflows.

**Outcome:**
Improved responsiveness and smoother handling of larger documents.

---

### 5. UI / UX Refinement

**The Challenge:**
As features expanded, chat history and debug outputs made the interface cluttered.

**The Solution:**

* Moved chat history to the sidebar.
* Improved debug mode formatting.
* Added response timing and cleaner controls.

**Outcome:**
Cleaner user experience and more professional interface presentation.

## 🖥️ How to Run

```bash
git clone https://github.com/amirthaakarthika-star/MULTICLOUD-RAG-AI-PDF-INTELLIGENCE.git
cd MULTICLOUD-RAG-AI-PDF-INTELLIGENCE
pip install -r requirements.txt
## 📸 Screenshots

### 🖥️ UI
![UI](ui.png)

### ⚙️ Processing Documents
![Processing](processing.png)

### 🧠 AI Answer Output
![Answer](answer.png)

### 🔍 Debug View (Explainable AI)
![Debug](debug.png)

### 🧠 Chat History
![History](screenshots/history.png)

### 📄 PDF Download
![PDF](screenshots/pdf.png)


