# DebugML - ML Failure Mode Search using Endee (Semantic Search + RAG)

## Project Overview and Problem Statement

Machine Learning experiments fail frequently due to issues such as overfitting, unstable training, data imbalance, or poor hyperparameter choices.  
However, the knowledge gained from these failed experiments is often lost, leading to repeated mistakes and wasted effort.

This project builds a **Semantic Search + Retrieval-Augmented Generation (RAG)** system that allows ML practitioners to:
- Search past ML experiment failures using natural language
- Retrieve semantically similar failure cases
- Generate actionable insights and fixes using a local LLM

The system is designed around **Endee**, a high-performance vector database, to act as a long-term semantic memory for ML failures.

---

## System Design and Technical Approach

### High-Level Architecture
```bash
User (Streamlit UI)
      ↓
Text Embedding (SentenceTransformers)
      ↓
Endee Vector Database (Similarity Search)
      ↓
Retrieved Failure Cases
      ↓
LLM (Ollama – LLaMA)
      ↓
RAG-based Explanation & Fix Suggestions
```
---
## Key Components
Streamlit UI: A clean interface for natural language querying.

Embedding Layer: Uses sentence-transformers/all-MiniLM-L6-v2 to convert descriptions into dense vectors.

Endee Vector Database: Manages historical experiment failures with high-speed similarity search.

RAG Layer (Ollama – LLaMA): Uses retrieved context to generate human-readable recommendations.

## 🧠 Why Endee?
Endee serves as the vector memory layer of the system. Each failure is stored with its description, observed symptoms, and attempted fixes.

Performance: Optimized for large-scale similarity queries.

Persistence: Acts as a long-term semantic memory for ML workflows.

Separation of Concerns: Clearly decouples vector storage (Endee) from reasoning (LLM).

## 🛠️ Setup & Execution
1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd <repo-name>
```
2. Environment Setup
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```
3. Run Endee
Endee should be running at http://localhost:8080. You can deploy it via the official Endee Labs Documentation using:
```bash
Native: install.sh and run.sh

Manual: Build via CMake

Container: Docker / Docker Compose
```

4. Ingest ML Failure Data
Populate the vector database with historical failure cases:
```bash
python ingest.py
```
5. Launch the App
```bash
streamlit run app.py
```
Visit http://localhost:8501 to start searching.

## 📝 Example Queries
"Model overfits after increasing depth"

"Training becomes unstable with high learning rate"

"Loss explodes after a few epochs"

## 🔮 Future Work
Automatic Clustering: Identify recurring failure patterns across different projects.

Time-Aware Memory: Prioritize more recent experiment failures.

Tool Integration: Deeper integration with MLflow or Weights & Biases to automatically ingest failed runs.
