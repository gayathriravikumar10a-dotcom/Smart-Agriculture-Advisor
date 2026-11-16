
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from pypdf import PdfReader

from rag.embedding_utils import generate_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "rag" / "documents"
DATABASE_DIR = PROJECT_ROOT / "rag" / "vector_store"
DATABASE_PATH = DATABASE_DIR / "agriculture_knowledge.sqlite3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 10


def create_database():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER,
                chunk_number INTEGER,
                document_hash TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        """)


def split_into_chunks(text):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def read_documents():
    all_chunks = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in rag/documents.")
        return []

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        file_hash = hashlib.sha256(
            pdf_path.read_bytes()
        ).hexdigest()

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                continue

            page_chunks = split_into_chunks(text)

            for chunk_number, content in enumerate(page_chunks, start=1):
                chunk_id = hashlib.sha256(
                    f"{file_hash}:{page_number}:{chunk_number}".encode()
                ).hexdigest()

                all_chunks.append({
                    "id": chunk_id,
                    "content": content,
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk_number": chunk_number,
                    "document_hash": file_hash,
                })

        print(f"Pages read: {len(reader.pages)}")

    return all_chunks


def ingest_documents():
    create_database()
    chunks = read_documents()
    if not chunks:
        print("No document text available to ingest.")
        return

    print(f"\nTotal chunks created: {len(chunks)}")

    with sqlite3.connect(DATABASE_PATH) as connection:
        completed = {
            row[0]
            for row in connection.execute("SELECT id FROM chunks")
        }

    pending = [
        item for item in chunks
        if item["id"] not in completed
    ]
    print(f"Already stored: {len(completed)}")
    print(f"Remaining chunks: {len(pending)}")
    if not pending:
        print("All document chunks are already stored!")
        return

    print("\nGenerating Gemini embeddings...")
    records = []

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        texts = [item["content"] for item in batch]

        while True:
            try:
                embeddings = generate_embeddings(
                    texts,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                break
            except Exception as error:
                message = str(error)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    print("\nGemini quota reached.")
                    print("Waiting 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    raise

        for item, embedding in zip(batch, embeddings):
            records.append((
                item["id"],
                item["content"],
                item["source"],
                item["page"],
                item["chunk_number"],
                item["document_hash"],
                json.dumps(embedding),
            ))

        print(
            f"Embedded {min(start + BATCH_SIZE, len(pending))}"
            f" / {len(pending)} remaining chunks"
        )

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.executemany("""
            INSERT OR REPLACE INTO chunks
            (id, content, source, page, chunk_number,
             document_hash, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)

    print("\nPDF ingestion completed successfully!")
    print(f"Stored {len(records)} chunks.")
    print(f"Database: {DATABASE_PATH}")


if __name__ == "__main__":
    ingest_documents()
