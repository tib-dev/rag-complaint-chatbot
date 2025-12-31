


```text
rag-complaint-chatbot/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # CI pipeline: lint, test, build
│       └── lint-and-tests.yml       # Additional checks
│
├── config/                          # YAML configuration files
│   ├── paths.yaml                   # Data paths (raw/interim/processed)
│   ├── preprocessing.yaml           # Text cleaning rules
│   ├── embedding.yaml               # Model name, chunk size/overlap
│   ├── vectorstore.yaml             # FAISS / ChromaDB settings
│   ├── rag.yaml                     # Retriever k, prompt templates
│   └── ui.yaml                      # Streamlit/Gradio UI settings
│
├── data/                             # All datasets
│   ├── raw/                          # Original CFPB complaints (tracked by DVC)
│   │   └── cfpb_complaints.csv
│   ├── interim/                      # Filtered/cleaned data
│   ├── processed/                    # Chunked & embedded data
│   ├── samples/                      # Stratified samples (~10-15K)
│   └── external/                     # Pre-built embeddings (tracked by DVC)
│       └── complaint_embeddings.parquet
│
├── vector_store/                      # Persisted vector indexes
│   ├── chroma/                        # ChromaDB vector store
│   └── faiss/                         # FAISS vector store
│
├── notebooks/                         # Jupyter notebooks
│   ├── eda/                           # Task 1: Exploratory Data Analysis
│   ├── preprocessing/                 # Task 1: text cleaning experiments
│   ├── embeddings/                    # Task 2: chunking & embedding experiments
│   └── evaluation/                    # Task 3: RAG qualitative evaluation
│
├── scripts/                           # CLI entry points for each pipeline
│   ├── run_eda.py
│   ├── run_preprocessing.py
│   ├── build_sample_embeddings.py
│   ├── build_vectorstore.py
│   ├── evaluate_rag.py
│   └── launch_ui.py
│
├── src/
│   └── rag_chatbot/
│       ├── __init__.py
│
│       ├── core/                      # Global settings & config loader
│       │   ├── __init__.py
│       │   └── settings.py
│
│       ├── data/                      # Loading and filtering raw/interim data
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── filter.py
│
│       ├── preprocessing/             # Text cleaning & normalization
│       │   ├── __init__.py
│       │   ├── cleaner.py
│       │   └── boilerplate.py
│
│       ├── chunking/                  # Text chunking logic
│       │   ├── __init__.py
│       │   └── text_splitter.py
│
│       ├── embeddings/                # Embedding generation
│       │   ├── __init__.py
│       │   └── embedder.py
│
│       ├── vectorstore/               # FAISS / ChromaDB wrappers
│       │   ├── __init__.py
│       │   ├── chroma.py
│       │   └── faiss.py
│
│       ├── rag/                       # RAG pipeline (retrieval + generation)
│       │   ├── __init__.py
│       │   ├── retriever.py
│       │   ├── prompt.py
│       │   ├── generator.py
│       │   └── pipeline.py
│
│       ├── evaluation/                # Qualitative evaluation logic
│       │   ├── __init__.py
│       │   └── qualitative.py
│
│       ├── ui/                        # Gradio / Streamlit interface
│       │   ├── __init__.py
│       │   ├── app_gradio.py
│       │   └── app_streamlit.py
│
│       └── utils/                     # Shared utilities
│           ├── __init__.py
│           ├── logging.py
│           ├── text.py
│           └── timing.py
│
├── tests/                             # Automated tests
│   ├── unit/
│   │   ├── test_cleaner.py
│   │   ├── test_chunking.py
│   │   ├── test_embeddings.py
│   │   └── test_retriever.py
│   └── integration/
│       └── test_rag_pipeline.py
│
├── docker/                             # Containerization
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── start.sh
│
├── dvc/                                # Optional DVC stages/config
├── .dvc/                               # DVC internal metadata
├── dvc.yaml                             # DVC pipeline definition
├── params.yaml                          # Global pipeline params
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env
├── README.md
└── .gitignore
```