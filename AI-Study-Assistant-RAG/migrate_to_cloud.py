import os
import chromadb
from dotenv import load_dotenv

load_dotenv(override=True)

COLLECTION_NAME = "study_documents"

cloud_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

collection = cloud_client.get_collection(
    name=COLLECTION_NAME
)

total = collection.count()

print(f"Cloud collection contains {total} records.")

if total == 0:
    print("No records found.")
    exit()

result = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

ids = result["ids"]
documents = result["documents"]
metadatas = result["metadatas"]
embeddings = result["embeddings"]

new_metadatas = []

for i, metadata in enumerate(metadatas):
    new_metadatas.append({
        "filename": "sample_research_paper.pdf",
        "chunk": i + 1
    })

collection.update(
    ids=ids,
    metadatas=new_metadatas
)

print("Metadata updated successfully!")

print(
    "Cloud collection count:",
    collection.count()
)