import os
import json
import boto3
import time
from pypdf import PdfReader
from dotenv import load_dotenv

# ================================================================
# LOAD ENV
# ================================================================
load_dotenv(dotenv_path=".env")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

# ================================================================
# AWS BEDROCK CLIENT
# ================================================================
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-southeast-2"
)

# ================================================================
# 1. LOAD PDFs
# ================================================================
def load_pdfs(data_folder="data"):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_folder, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documents.append({
                "source": filename,
                "text": text.strip()
            })
            print(f"✅ Loaded: {filename}")
    return documents

# ================================================================
# 2. SPLIT TEXT INTO RAW CHUNKS
# ================================================================
def split_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ================================================================
# 3. ENRICH EACH CHUNK USING BEDROCK NOVA
# ================================================================
def enrich_chunk(chunk_text, source):
    prompt = f"""
You are an intelligent document analyst for Metricon Homes.
Analyze the following text chunk and return a JSON object with these exact fields:
- title: A short descriptive title (max 10 words)
- summary: A concise summary (max 50 words)
- keywords: A list of 5 important keywords
- category: One of [Building Process, Costs & Finance, Why Metricon, FAQ, General]
- importance: A score from 1-10 indicating usefulness for customer queries

Text chunk:
{chunk_text}

Respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.
"""
    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 300,
                    "temperature": 0.2
                }
            })
        )
        output = json.loads(response["body"].read())
        raw = output["output"]["message"]["content"][0]["text"]
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        enriched = json.loads(raw)
        return enriched

    except Exception as e:
        print(f"⚠️ Enrichment failed: {e}")
        return {
            "title": "General Information",
            "summary": chunk_text[:100],
            "keywords": [],
            "category": "General",
            "importance": 5
        }

# ================================================================
# 4. FULL AGENTIC CHUNKING PIPELINE
# ================================================================
def agentic_chunking_pipeline(data_folder="data"):
    documents = load_pdfs(data_folder)
    all_chunks = []
    chunk_id = 0

    for doc in documents:
        print(f"\n📄 Processing: {doc['source']}")
        raw_chunks = split_text(doc["text"])
        print(f"   → {len(raw_chunks)} raw chunks found")

        for i, chunk in enumerate(raw_chunks):
            print(f"   🧠 Enriching chunk {i+1}/{len(raw_chunks)}...")
            enriched = enrich_chunk(chunk, doc["source"])

            all_chunks.append({
                "id": chunk_id,
                "source": doc["source"],
                "text": chunk,
                "title": enriched.get("title", ""),
                "summary": enriched.get("summary", ""),
                "keywords": enriched.get("keywords", []),
                "category": enriched.get("category", "General"),
                "importance": enriched.get("importance", 5)
            })

            chunk_id += 1
            time.sleep(0.5)

    # Save to JSON
    output_path = "chunker/chunks.json"
    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n✅ Done! {len(all_chunks)} enriched chunks saved to {output_path}")
    return all_chunks

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    agentic_chunking_pipeline()