import fitz
import pickle
from pathlib import Path

from src.config import PDF_DIR, DOCS_PATH


def load_pdf(pdf_path: Path):
    pages = []

    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "metadata": {
                    "filename": pdf_path.name,
                    "page": page_number
                }
            })

    return pages


def load_all_pdfs(pdf_dir: Path):
    all_pages = []

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir}. Put PDFs inside data/pdfs/"
        )

    for pdf_path in pdf_paths:
        print(f"Loading: {pdf_path.name}")
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    return all_pages


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


def chunk_documents(pages):
    documents = []

    for page in pages:
        chunks = chunk_text(page["text"])

        for chunk_id, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "metadata": {
                    "filename": page["metadata"]["filename"],
                    "page": page["metadata"]["page"],
                    "chunk_id": chunk_id
                }
            })

    return documents


def build_documents():
    pages = load_all_pdfs(PDF_DIR)
    documents = chunk_documents(pages)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print(f"Saved {len(documents)} chunks to {DOCS_PATH}")

    return documents


if __name__ == "__main__":
    build_documents()