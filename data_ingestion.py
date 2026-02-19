import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# ================================================================
# LOAD EMBEDDING MODEL
# ================================================================
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded!")

# ================================================================
# CONNECT TO QDRANT
# ================================================================
qdrant = QdrantClient(url="http://localhost:6333")
COLLECTION = "metricon_rag"

# ================================================================
# CREATE COLLECTION
# ================================================================
def create_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        qdrant.delete_collection(COLLECTION)
        print(f"🗑️ Deleted existing collection: {COLLECTION}")

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"✅ Created collection: {COLLECTION}")

# ================================================================
# INGEST CHUNKS
# ================================================================
def ingest_chunks():
    with open("chunker/chunks.json", "r") as f:
        chunks = json.load(f)

    print(f"📦 Ingesting {len(chunks)} chunks into Qdrant...")

    points = []
    for chunk in chunks:
        text_to_embed = f"{chunk['title']} {chunk['summary']} {chunk['text']}"
        vector = embedder.encode(text_to_embed).tolist()

        points.append(PointStruct(
            id=chunk["id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                "title": chunk["title"],
                "summary": chunk["summary"],
                "keywords": chunk["keywords"],
                "category": chunk["category"],
                "importance": chunk["importance"],
                "source": chunk["source"]
            }
        ))

    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"✅ Successfully ingested {len(points)} chunks!")

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    create_collection()
    ingest_chunks()
    print("\n🎉 Data ingestion complete! Qdrant is ready.")