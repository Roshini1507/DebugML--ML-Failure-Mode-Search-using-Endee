import requests
import json
from sentence_transformers import SentenceTransformer

ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "ml_failure_memory"

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_failures(query, k=3):
    embedding = model.encode(query).tolist()

    payload = {
        "index_name": INDEX_NAME,
        "vector": embedding,
        "top_k": k
    }

    response = requests.post(
        f"{ENDEE_URL}/api/v1/index/search",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Endee search failed ({response.status_code})\\n{response.text}"
        )

    data = response.json()

    # Expected Endee format:
    # { "results": [ { "id": ..., "score": ..., "metadata": {...} }, ... ] }
    return data["results"]
