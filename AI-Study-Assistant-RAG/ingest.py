import os
import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

PDF_FOLDER = "./uploaded_pdfs"
COLLECTION_NAME = "study_documents"

# Chroma Cloud
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# Embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metadatas = []
ids = []

pdf_files = [
    file for file in os.listdir(PDF_FOLDER)
    if file.lower().endswith(".pdf")
]

if not pdf_files:
    print("No PDF found in uploaded_pdfs.")
    exit()

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)

    print(f"\nReading: {pdf_file}")

    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Split text into chunks
    chunk_size = 1000

    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size].strip()

        if chunk:
            documents.append(chunk)
            metadatas.append({
                "source": pdf_file
            })
            ids.append(
                f"{pdf_file}_{i}"
            )

print(f"\nTotal chunks: {len(documents)}")

if not documents:
    print("No readable text found in PDF.")
    exit()

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
).tolist()

print("Uploading to Chroma Cloud...")

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("\nIngestion completed successfully!")
print("Cloud collection count:", collection.count())