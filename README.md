
# RAG Complaint Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG) chatbot** for consumer complaints.  
The system processes raw CFPB complaint data, builds embeddings, and provides natural language responses via a RAG pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Objectives](#objectives)
- [Dataset Overview](#dataset-overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Pipeline & Modeling Approach](#pipeline--modeling-approach)
- [MLOps & Engineering Practices](#mlops--engineering-practices)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## Project Overview

This project implements a **RAG chatbot** for consumer complaints.  
It enables efficient information retrieval from large complaint datasets while generating coherent answers in natural language.

The system covers:

- Data ingestion and cleaning of CFPB complaint datasets
- Text chunking and embedding generation
- FAISS/Chroma vector store construction
- RAG pipeline for retrieval + generation
- Qualitative evaluation of responses
- Streamlit/Gradio user interface
- Containerized deployment with Docker
- Data versioning and reproducibility with DVC

---

## Business Context

Financial institutions and consumer protection agencies need fast, accurate responses to complaints.  
Manual review is slow, error-prone, and expensive.

This chatbot supports:

- **Automated complaint responses**
- **Insight extraction from large datasets**
- **Research and monitoring of consumer issues**
- **Enhanced customer support efficiency**

---

## Objectives

- Build a retrieval-augmented system for complaint data
- Generate accurate and context-aware responses
- Maintain a scalable vector store for embeddings
- Ensure reproducibility with DVC and versioned datasets
- Deploy via API or interactive UI

---

## Dataset Overview

**Source:** CFPB consumer complaints  

**Key fields:**

| Column          | Description                        |
| --------------- | ---------------------------------- |
| ComplaintID     | Unique complaint identifier        |
| Product         | Complaint category                 |
| Issue           | Detailed complaint issue           |
| ConsumerComplaint | Complaint narrative               |
| Company         | Company involved                   |
| DateReceived    | Complaint submission timestamp     |

Derived features:

- Cleaned and normalized complaint text
- Chunked complaint segments for embedding
- Metadata for retrieval and filtering

---

## Project Structure

```text
rag-complaint-chatbot/
│
├── config/                      # YAML configs: paths, embeddings, RAG, UI
├── data/                        # Raw, interim, processed, external datasets
├── vector_store/                # Persisted FAISS / Chroma embeddings
├── notebooks/                   # EDA, preprocessing, embeddings, evaluation
├── scripts/                     # CLI scripts to run pipelines and UI
├── src/
│   └── rag_chatbot/             # Main Python package
│       ├── core/                # Settings and configuration loader
│       ├── data/                # Data loading and filtering
│       ├── preprocessing/       # Text cleaning & normalization
│       ├── chunking/            # Text chunking logic
│       ├── embeddings/          # Embedding generation
│       ├── vectorstore/         # FAISS / ChromaDB wrappers
│       ├── rag/                 # RAG pipeline (retriever + generator)
│       ├── evaluation/          # Qualitative evaluation logic
│       ├── ui/                  # Streamlit / Gradio interface
│       └── utils/               # Shared utilities
├── tests/                       # Unit & integration tests
├── docker/                       # Dockerfile and compose
├── dvc.yaml                       # DVC pipeline
├── params.yaml                    # Global parameters
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Architecture

```text
+---------------------------+
| Raw Complaint Data        |
| - CFPB complaints CSV     |
| Location: data/raw/       |
+-----------+---------------+
            |
            v
+---------------------------+
| Data Loader               |
| src/rag_chatbot/data      |
+-----------+---------------+
            |
            v
+---------------------------+
| Preprocessing             |
| - Text cleaning           |
| - Normalization           |
+-----------+---------------+
            |
            v
+---------------------------+
| Chunking & Embeddings     |
| - TextSplitter            |
| - Embeddings via OpenAI   |
+-----------+---------------+
            |
            v
+---------------------------+
| Vector Store              |
| - FAISS / Chroma          |
+-----------+---------------+
            |
            v
+---------------------------+
| RAG Pipeline              |
| - Retriever               |
| - Generator               |
+-----------+---------------+
            |
            v
+---------------------------+
| UI / API                  |
| - Gradio / Streamlit      |
| - FastAPI for production  |
+---------------------------+
```

---

## Pipeline & Modeling Approach

1. **EDA** – Explore distribution of complaint categories, narrative length, missing values.
2. **Preprocessing** – Text cleaning: lowercasing, punctuation removal, stopword handling.
3. **Chunking** – Split long complaint narratives into manageable text chunks.
4. **Embedding** – Generate vector representations for each chunk.
5. **Vector Store** – Store embeddings in FAISS or Chroma for efficient retrieval.
6. **RAG Pipeline** – Combine retrieval and generation to answer queries.
7. **Evaluation** – Qualitative analysis of responses against test complaints.
8. **UI / API** – Interactive interface for testing the chatbot or integrating into systems.

---

## MLOps & Engineering Practices

- **Data Versioning:** DVC tracks raw and processed datasets, embeddings, and vector stores
- **Experiment Reproducibility:** Parameters stored in `params.yaml`
- **Containerization:** Docker & docker-compose for easy deployment
- **Testing:** Unit and integration tests for preprocessing, chunking, embedding, and RAG pipeline

---

## Setup & Installation

Clone the repository:

```bash
git clone https://github.com/<username>/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install in editable mode:

```bash
pip install -e .
```

---

## Running the Project

### Run full pipeline (DVC)

```bash
dvc repro
```

### Launch UI (Gradio / Streamlit)

```bash
python scripts/launch_ui.py
```

### Docker

Build and run container:

```bash
docker-compose up --build
```

---

## Technologies Used

- Python 3.10+
- OpenAI API / LangChain
- FAISS / Chroma
- DVC (data version control)
- FastAPI
- Streamlit / Gradio
- Docker & docker-compose
- Pytest for testing

---

## Author

Tibebu Kaleb – Full-stack AI/ML engineer with experience in NLP and RAG pipelines

