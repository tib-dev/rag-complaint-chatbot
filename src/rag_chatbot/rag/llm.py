import os
from huggingface_hub import hf_hub_download
from langchain_community.llms import CTransformers
from rag_chatbot.core.settings import settings

REPO_ID = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

MODEL_DIR = settings.paths.MODEL["model_dir"]

_LLM = None


def get_llm():
    global _LLM

    if _LLM is not None:
        return _LLM

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"Loading Mistral via CTransformers from: {model_path}")

    # CTransformers config mapping for your parameters
# Optimized config for TinyLlama on CPU
    config = {
        'max_new_tokens': 256,    # Shorter answers = faster finish
        'temperature': 0.0,       # Deterministic for RAG
        'repetition_penalty': 1.1,
        'context_length': 1024,   # TinyLlama handles 2048, but 1024 is faster
        'threads': 4,             # Ensure this matches your physical cores
        'batch_size': 128,        # Increased from 32 to process prompt faster
        'stream': True            # Essential for perception of speed
    }

    _LLM = CTransformers(
        model=model_path,
        model_type="llama",      
        config=config
    )

    return _LLM
