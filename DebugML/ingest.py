import json
import requests
from sentence_transformers import SentenceTransformer

ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "ml_failure_memory"
VECTOR_DIM = 384  # all-MiniLM-L6-v2

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- 1. CREATE INDEX ----------
# ---------- 1. CREATE INDEX ----------
create_payload = {
    "index_name": INDEX_NAME,
    "schema": {
        "vector_dim": VECTOR_DIM,
        "distance": "cosine"
    },
    "index_type": "flat"
}

create_resp = requests.post(
    f"{ENDEE_URL}/api/v1/index/create",
    headers={"Content-Type": "application/json"},
    json=create_payload
)

if create_resp.status_code not in (200, 201, 409):
    raise RuntimeError(
        f"Index creation failed ({create_resp.status_code})\n{create_resp.text}"
    )

print("✔ Index created or already exists")


# ---------- 2. LOAD DATA ----------
with open("data/experiments.json") as f:
    experiments = json.load(f)

vectors = []
for exp in experiments:
    embedding = model.encode(exp["text"]).tolist()
    vectors.append({
        "id": exp["id"],
        "vector": embedding,
        "metadata": exp
    })

# ---------- 3. INSERT VECTORS ----------
insert_resp = requests.post(
    f"{ENDEE_URL}/api/v1/index/insert",
    headers={"Content-Type": "application/json"},
    json={
        "index_name": INDEX_NAME,
        "vectors": vectors
    }
)

if insert_resp.status_code != 200:
    raise RuntimeError(
        f"Insert failed ({insert_resp.status_code})\n{insert_resp.text}"
    )

print("✔ Experiments ingested into Endee")
