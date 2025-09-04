import pdfplumber

def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber (reasonable handling of typical PDFs)."""
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text()
            except Exception:
                text = None
            if text:
                texts.append(text)
    return "\n".join(texts)

def simple_clean(text: str) -> str:
    # replace multiple newlines with one
    out = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    # normalize whitespace
    out = " ".join(out.split())
    return out

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks
