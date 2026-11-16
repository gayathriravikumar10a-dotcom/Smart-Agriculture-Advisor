
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "rag" / "documents"
VECTOR_STORE_DIR = PROJECT_ROOT / "rag" / "vector_store"

COLLECTION_NAME = "agriculture_knowledge"

# Embedding model converts text into numerical vectors.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent ChromaDB storage.
client = chromadb.PersistentClient(
    path=str(VECTOR_STORE_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from an agricultural PDF."""

    reader = PdfReader(str(pdf_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n".join(pages)


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks."""

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def ingest_documents() -> int:
    """Read PDFs, create embeddings and store chunks."""

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF documents found in rag/documents.")
        return 0

    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        text = extract_pdf_text(pdf_path)
        chunks = split_into_chunks(text)

        if not chunks:
            print("No readable text found.")
            continue

        ids = [
            f"{pdf_path.stem}_{index}"
            for index in range(len(chunks))
        ]

        embeddings = embedding_model.encode(
            chunks
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": pdf_path.name}
                for _ in chunks
            ],
        )

        total_chunks += len(chunks)

        print(f"Stored {len(chunks)} chunks.")

    print(f"\nTotal chunks stored: {total_chunks}")

    return total_chunks


if __name__ == "__main__":
    ingest_documents()

