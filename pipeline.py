from safety_llm_pipeline import demo_pipeline
import numpy as np

def pipeline(pdf_folder: str, process_normal_data: np.ndarray):
    retriever = Retriever(embed_model_name="all-MiniLM-L6-v2")
    n_added = build_corpus_from_pdf_folder(pdf_folder, retriever)
    print(f"Added {n_added} text chunks to retriever (docstore size={len(retriever.docstore)})")


if __name__ == "__main__":
    PDF_FOLDER = "safety_pdfs"

