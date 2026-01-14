import gradio as gr
from typing import List, Dict
from rag_chatbot.rag.pipeline import RAGPipeline


def format_sources(sources: List[Dict]) -> str:
    """Format retrieved documents for display."""
    if not sources:
        return "No supporting documents found."

    lines = ["## Supporting Documents\n"]
    for i, s in enumerate(sources[:3], 1):
        lines.append(
            f"### Source {i}\n"
            f"- Complaint ID: {s.get('complaint_id', 'N/A')}\n"
            f"- Product: {s.get('product_category', 'N/A')}\n"
            f"- Issue: {s.get('issue', 'N/A')}\n\n"
            f"> {s.get('document', '').strip()}\n\n"
            f"Score: {round(s.get('score', 0.0), 4)}\n"
            "---"
        )
    return "\n".join(lines)


def launch_ui(rag: RAGPipeline):
    """
    Launch a modern Gradio UI compatible with version 6.0+.
    """

    def rag_chat(query: str):
        if not query.strip():
            return "Enter a question.", 0.0, ""

        try:
            result = rag.run(query)
            return (
                result.get("answer", "No answer generated."),
                result.get("confidence", 0.0),
                format_sources(result.get("sources", [])),
            )
        except Exception as e:
            return f"Error: {str(e)}", 0.0, ""

    # FIXED: Moved 'theme' from here to .launch()
    with gr.Blocks(title="CrediTrust Insight Engine") as demo:
        gr.Markdown(
            """
            # Complaint Analysis (RAG)
            Ask a question based on customer complaint narratives.
            Answers are generated only from retrieved documents.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Question",
                    placeholder="What are common issues in credit card billing?",
                    lines=3,
                )

                submit_btn = gr.Button("Run Analysis", variant="primary")
                clear_btn = gr.Button("Clear")

                answer_output = gr.Textbox(
                    label="Answer",
                    lines=8,
                    interactive=False,
                    # FIXED: Removed 'show_copy_button' (unsupported in v6.0)
                )

                confidence_output = gr.Slider(
                    label="Confidence",
                    minimum=0,
                    maximum=1,
                    value=0,
                    interactive=False,
                )

            with gr.Column(scale=1):
                sources_output = gr.Markdown(
                    label="Sources",
                    value="*Retrieved documents will appear here.*",
                )

        submit_btn.click(
            fn=rag_chat,
            inputs=query_input,
            outputs=[answer_output, confidence_output, sources_output],
        )

        # Corrected lambda to reset all fields
        clear_btn.click(
            fn=lambda: (
                "", 0.0, "*Retrieved documents will appear here.*", ""),
            outputs=[query_input, confidence_output,
                     sources_output, answer_output],
        )

    # FIXED: theme is now passed here in Gradio 6.0
    demo.launch(theme=gr.themes.Soft(), share=False)


if __name__ == "__main__":
    # rag = RAGPipeline(...)
    # launch_ui(rag)
    pass
