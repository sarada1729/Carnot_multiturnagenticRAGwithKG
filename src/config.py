from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"

STORAGE_DIR = ROOT_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "faiss.index"
DOCS_PATH = STORAGE_DIR / "documents.pkl"
KG_PATH = STORAGE_DIR / "kg.pkl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_NAME = "gemma3:1b"