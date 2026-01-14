from langchain_core.prompts import PromptTemplate


def get_prompt():
    return PromptTemplate(
        template="""<|system|>
You are a factual financial assistant. Use ONLY the provided context. 
If the information is not there, say "I do not have enough information to answer this question."<|user|>
Context:
{context}

Question: {question}<|assistant|>
""",
        input_variables=["context", "question"],
    )
