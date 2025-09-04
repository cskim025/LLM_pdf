import os
import json
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2", embedding_dim: int = None):
        self.embed_model = SentenceTransformer(embed_model_name)
        if embedding_dim is None:
            # get dimension from model
            self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # cosine via normalized vectors
        self.docstore: List[Dict[str, Any]] = []  # parallel store for metadata and text

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalize for cosine similarity with IndexFlatIP
        faiss.normalize_L2(embs)
        return embs

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        assert len(texts) == len(metadatas)
        embs = self._embed(texts)
        self.index.add(embs)
        for text, md in zip(texts, metadatas):
            self.docstore.append({"text": text, "meta": md})

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        faiss.write_index(self.index, path_prefix + ".faiss")
        with open(path_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(self.docstore, f, ensure_ascii=False, indent=2)

    def load(self, path_prefix: str):
        self.index = faiss.read_index(path_prefix + ".faiss")
        with open(path_prefix + ".json", "r", encoding="utf-8") as f:
            self.docstore = json.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        q_emb = self._embed([query])[0].astype("float32")
        D, I = self.index.search(np.array([q_emb]), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.docstore):
                continue
            results.append((self.docstore[idx], float(score)))
        return results
