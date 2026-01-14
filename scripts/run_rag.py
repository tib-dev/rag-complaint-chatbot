from rag_chatbot.rag.retriever import Retriever
from rag_chatbot.rag.query_embedder import QueryEmbedder
from rag_chatbot.rag.llm import get_llm
from rag_chatbot.prompt.prompts import get_prompt
from rag_chatbot.rag.pipeline import RAGPipeline
from rag_chatbot.core.settings import settings
from rag_chatbot.ui.app_gradio import launch_ui

persist_path = settings.paths.VECTOR_STORE["fiass_dir"]

rag = RAGPipeline(
    embedder=QueryEmbedder(),
    retriever=Retriever(
        persist_path / "faiss.index",
        persist_path / "metadata.parquet",
    ),
    llm=get_llm(),
    prompt=get_prompt(),
)

while True:
    q = input("\nAsk a question (or 'exit'): ")
    if q.lower() == "exit":
        break
    print(rag.run(q)["answer"])


launch_ui(rag)