import asyncio
from typing import Dict, Any, List
from rag_chatbot.rag.hallucination_guard import should_answer
from rag_chatbot.rag.confidence import compute_confidence


class RAGPipeline:
    def __init__(self, embedder, retriever, llm, prompt):
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    def _format_context(self, chunks: List[Dict]) -> str:
        """Cleans and formats retrieved chunks into a single string."""
        # Remove extra newlines/whitespace to save CPU processing tokens
        docs = [c["document"].strip() for c in chunks]
        return "\n\n---\n\n".join(docs)

    async def arun(self, query: str) -> Dict[str, Any]:
        """Asynchronous execution for better performance in web/app environments."""

        # 1. Retrieval
        query_emb = self.embedder.embed(query)
        retrieved_chunks = self.retriever.retrieve(query_emb)

        # 2. Guardrails (Hard Block)
        if not should_answer(retrieved_chunks):
            return {
                "query": query,
                "answer": "I'm sorry, I don't have enough information in my database to answer that accurately.",
                "confidence": 0.0,
                "sources": [],
            }

        # 3. Context Preparation
        context = "\n\n".join(c["document"] for c in retrieved_chunks[:2])
        context = context[:2000]

        # 4. Prompt Construction
        # Using LCEL style formatting
        formatted_prompt = self.prompt.format(
            context=context,
            question=query
        )

        # 5. Generation (Async)
        # We use ainvoke to allow other tasks to run while the CPU "thinks"
        try:
            answer = await self.llm.ainvoke(formatted_prompt)
            # Handle if answer is a BaseMessage (LangChain standard)
            if hasattr(answer, "content"):
                answer = answer.content
        except Exception as e:
            answer = f"Error during generation: {str(e)}"

        # 6. Post-processing
        confidence = compute_confidence(retrieved_chunks)

        return {
            "query": query,
            "answer": answer,
            "confidence": round(float(confidence), 2),
            "sources": retrieved_chunks,
        }

    def run(self, query: str) -> Dict[str, Any]:
        """Synchronous wrapper for the async run."""
        return asyncio.run(self.arun(query))
