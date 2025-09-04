from text_embedding import Retriever
from merged import build_corpus_from_pdf_folder, run_query_and_suggest

def pipeline(pdf_folder: str, process_normal_data: np.ndarray):
    retriever = Retriever(embed_model_name="all-MiniLM-L6-v2")
    n_added = build_corpus_from_pdf_folder(pdf_folder, retriever)
    print(f"Added {n_added} text chunks to retriever (docstore size={len(retriever.docstore)})")

    problem_description = "Two Liquids (A and B) at plant X are mixed at room temperature and pressure. Need phase separation information."
    suggestion = run_query_and_suggest(problem_description, retriever, generation_backend="openai")
    
if __name__ == "__main__":
    PDF_FOLDER = "safety_pdfs"

