import os
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq


# Load environment variables
load_dotenv(override=True)


# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "uploaded_pdfs"

DATA_DIR.mkdir(exist_ok=True)


# FastAPI application
app = FastAPI(
    title="AI Study Assistant using RAG",
    version="1.0.0"
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = None


def get_embedder():
    global embedding_model

    if embedding_model is None:
        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return embedding_model


# ============================================================
# CHROMADB CLOUD
# ============================================================

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)


collection = chroma_client.get_or_create_collection(
    name="study_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150
) -> List[str]:

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        # Prefer sentence/word boundary
        if end < len(text):

            boundary = max(
                text.rfind(". ", start, end),
                text.rfind(" ", start, end)
            )

            if boundary > start + int(chunk_size * 0.6):
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        ROOT / "static" / "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    metadata = collection.get(
        include=["metadatas"]
    ).get("metadatas", [])

    document_names = set()

    for metadata_item in metadata:

        if metadata_item:
            filename = metadata_item.get(
                "filename",
                ""
            )

            if filename:
                document_names.add(filename)

    return {
        "status": "online",
        "documents": len(document_names),
        "chunks": collection.count()
    }


# ============================================================
# DOCUMENT LIST
# ============================================================

@app.get("/documents")
def documents():

    result = collection.get(
        include=["metadatas"]
    )

    metadatas = result.get(
        "metadatas"
    ) or []

    names = sorted(
        set(
            metadata_item.get("filename", "")
            for metadata_item in metadatas
            if metadata_item
        )
    )

    names = [
        name
        for name in names
        if name
    ]

    return {
        "documents": names,
        "count": len(names),
        "chunks": collection.count()
    }


# ============================================================
# PDF UPLOAD
# ============================================================

@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...)
):

    if not files:

        raise HTTPException(
            400,
            "Please select at least one PDF."
        )

    added = []
    skipped = []
    errors = []

    for file in files:

        filename = Path(
            file.filename or "document.pdf"
        ).name

        # Only PDF files
        if not filename.lower().endswith(".pdf"):

            errors.append(
                f"{filename}: only PDF files are supported"
            )

            continue

        content = await file.read()

        if not content:

            errors.append(
                f"{filename}: file is empty"
            )

            continue

        # Save uploaded PDF
        save_path = (
            DATA_DIR /
            f"{uuid.uuid4().hex}_{filename}"
        )

        save_path.write_bytes(content)

        try:

            # Read PDF
            reader = PdfReader(
                str(save_path)
            )

            pages = []

            for page in reader.pages:

                pages.append(
                    page.extract_text() or ""
                )

            full_text = "\n".join(
                pages
            ).strip()

            # Check text
            if not full_text:

                errors.append(
                    f"{filename}: no selectable text found "
                    "(scanned PDFs need OCR)"
                )

                save_path.unlink(
                    missing_ok=True
                )

                continue

            # Create chunks
            chunks = split_text(
                full_text
            )

            if not chunks:

                errors.append(
                    f"{filename}: no readable text found"
                )

                save_path.unlink(
                    missing_ok=True
                )

                continue

            # Remove previous copy
            old = collection.get(
                where={
                    "filename": filename
                },
                include=[]
            )

            if old.get("ids"):

                collection.delete(
                    ids=old["ids"]
                )

            # Create embeddings
            vectors = get_embedder().encode(
                chunks,
                normalize_embeddings=True
            ).tolist()

            # Create IDs
            ids = [
                f"{uuid.uuid4().hex}_{i}"
                for i in range(len(chunks))
            ]

            # Metadata
            metadatas = [
                {
                    "filename": filename,
                    "chunk": i + 1
                }
                for i in range(len(chunks))
            ]

            # Upload to Chroma Cloud
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=vectors,
                metadatas=metadatas
            )

            added.append(
                {
                    "filename": filename,
                    "pages": len(reader.pages),
                    "chunks": len(chunks)
                }
            )

        except Exception as exc:

            save_path.unlink(
                missing_ok=True
            )

            errors.append(
                f"{filename}: {str(exc)}"
            )

    return {
        "added": added,
        "skipped": skipped,
        "errors": errors
    }


# ============================================================
# ASK AI
# ============================================================

@app.post("/ask")
def ask(
    request: AskRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            400,
            "Please enter a question."
        )

    # Check documents
    if collection.count() == 0:

        raise HTTPException(
            400,
            "Upload your PDF chapters first."
        )

    # Groq API key
    api_key = os.getenv(
        "GROQ_API_KEY",
        ""
    ).strip()

    if (
        not api_key
        or api_key == "your_groq_api_key_here"
    ):

        raise HTTPException(
            503,
            "Groq API key missing. "
            "Add GROQ_API_KEY to your .env file."
        )

    # Create question embedding
    query_vector = get_embedder().encode(
        [question],
        normalize_embeddings=True
    ).tolist()[0]

    # Search Chroma Cloud
    matches = collection.query(
        query_embeddings=[query_vector],
        n_results=min(
            4,
            collection.count()
        )
    )

    docs = matches.get(
        "documents",
        [[]]
    )[0]

    metas = matches.get(
        "metadatas",
        [[]]
    )[0]

    distances = matches.get(
        "distances",
        [[]]
    )[0]

    context_parts = []
    sources = []

    for doc, meta, distance in zip(
        docs,
        metas,
        distances
    ):

        context_parts.append(
            f"[Source: "
            f"{meta.get('filename')} | "
            f"Chunk {meta.get('chunk')}]\n"
            f"{doc}"
        )

        sources.append(
            {
                "filename": meta.get(
                    "filename"
                ),
                "chunk": meta.get(
                    "chunk"
                ),
                "similarity": round(
                    1 - float(distance),
                    3
                )
            }
        )

    context = "\n\n".join(
        context_parts
    )

    # ========================================================
    # GROQ
    # ========================================================

    try:

        client = Groq(
            api_key=api_key
        )

        completion = (
            client.chat.completions.create(
                model=os.getenv(
                    "GROQ_MODEL",
                    "llama-3.3-70b-versatile"
                ),
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content":
                        "You are a helpful AI Study Assistant. "
                        "Answer using only the supplied PDF context. "
                        "Explain clearly in student-friendly language. "
                        "If the answer is not present in the context, "
                        "say that the uploaded chapters do not provide "
                        "enough information. "
                        "Do not invent facts. "
                        "Mention relevant source filenames in the answer."
                    },
                    {
                        "role": "user",
                        "content":
                        f"PDF CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{question}"
                    }
                ]
            )
        )

        answer = (
            completion
            .choices[0]
            .message
            .content
            or "No answer was returned."
        )

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as exc:

        raise HTTPException(
            502,
            f"Groq request failed: {str(exc)}"
        )


# ============================================================
# DELETE ALL DOCUMENTS
# ============================================================

@app.delete("/documents")
def clear_documents():

    global collection

    chroma_client.delete_collection(
        "study_documents"
    )

    collection = (
        chroma_client.get_or_create_collection(
            name="study_documents",
            metadata={
                "hnsw:space": "cosine"
            }
        )
    )

    return {
        "message":
        "All indexed document chunks have been removed."
    }