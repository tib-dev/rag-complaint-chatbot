import rag_chatbot.data.filter as fl
import rag_chatbot.preprocessing.cleaning as cl
from rag_chatbot.core.settings import settings
from rag_chatbot.data.handler import DataHandler
from rag_chatbot.data.validation import validate_rag_ready


def run_preprocessing_pipeline() -> None:
    """
    End-to-end preprocessing pipeline for RAG:
    - Load raw CFPB complaints
    - Normalize schema
    - Filter products
    - Drop empty narratives
    - Clean text for embeddings
    - Validate RAG readiness
    - Persist cleaned dataset
    """

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    cols_cfg = settings.get("columns")
    filters_cfg = settings.get("filters")

    column_mapping = cols_cfg["mapping"]
    required_columns = set(cols_cfg["required"])
    allowed_products = filters_cfg["allowed_product_categories"]

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    df_raw = DataHandler.from_registry(
        section="DATA",
        path_key="raw_dir",
        filename="complaints.csv",
    ).load()

    # ------------------------------------------------------------------
    # Step 1: Schema normalization
    # ------------------------------------------------------------------
    df = cl.clean_and_select_columns(
        df_raw,
        column_mapping=column_mapping,
        required_columns=required_columns,
    )

    # ------------------------------------------------------------------
    # Step 2: Product filtering
    # ------------------------------------------------------------------
    df = fl.filter_by_products(
        df,
        allowed_products=allowed_products,
        product_column="product_category",
    )

    # ------------------------------------------------------------------
    # Step 3: Drop empty narratives
    # ------------------------------------------------------------------
    df = fl.filter_non_empty_narratives(
        df,
        narrative_column="consumer_complaint_narrative",
    )

    # ------------------------------------------------------------------
    # Step 4: Text cleaning (embedding-safe)
    # ------------------------------------------------------------------
    df = cl.apply_text_cleaning(df)

    # ------------------------------------------------------------------
    # Step 5: Validation
    # ------------------------------------------------------------------
    validate_rag_ready(df)

    # ------------------------------------------------------------------
    # Step 6: Persist cleaned data
    # ------------------------------------------------------------------
    DataHandler.from_registry(
        section="DATA",
        path_key="interim_dir",
        filename="complaints_clean.parquet",
    ).save(df)


if __name__ == "__main__":
    run_preprocessing_pipeline()
