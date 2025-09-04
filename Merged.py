import os
import glob
import json
from typing import List, Tuple, Dict, Any
from PDF_parser import extract_text_from_pdf, simple_clean, chunk_text
from tqdm import tqdm

def build_corpus_from_pdf_folder(pdf_folder: str, retriever: Retriever, chunk_size=512, overlap=64):
    pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    all_text_chunks = []
    all_meta = []
    for p in tqdm(pdf_paths, desc="Parsing PDFs"):
        raw = extract_text_from_pdf(p)
        cleaned = simple_clean(raw)
        if not cleaned:
            continue
        chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            all_text_chunks.append(c)
            all_meta.append({"source": os.path.basename(p), "chunk_id": i})
    retriever.add_documents(all_text_chunks, all_meta)
    return len(all_text_chunks)

def run_query_and_suggest(
    query_description: str,
    retriever: Retriever,
    generation_backend: str = "openai",
    top_k: int = 5
) -> str:

    hits = retriever.retrieve(query_description, top_k=top_k)
    context_pieces = []
    for doc, score in hits:
        context_pieces.append(f"Source: {doc['meta'].get('source','unknown')}\nText: {doc['text'][:1000]}")  # truncate

    # Compose prompt for the generator
    prompt = "You are a thermodyanmics expert. 
              Given the separation process description and the relevant thermodyanmics principals excerpts below, suggest concise process design features and actionable steps. 
              Flag any assumptions and recommend human expert review.\n\n"
    prompt += "PROCESS DESCRIPTION:\n" + query_description + "\n\n"

    for i, piece in enumerate(context_pieces, 1):
        prompt += f"\n--- EXCERPT {i} ---\n{piece}\n"

    prompt += "\n\nPlease output: (1) Short recommended safety rules (bullet list), (2) Required immediate actions, (3) Suggested monitoring/threshold changes, (4) Explicit assumptions and (5) Recommended domain-expert checks.\n"

    if generation_backend == "openai":
        return generate_with_openai(prompt, max_tokens=500)
    else:
        # fallback to HF
        return generate_with_hf(prompt, model_name=generation_backend, max_new_tokens=512)
