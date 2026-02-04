
# ML Failure Mode Search using Endee (Semantic Search + RAG)

## 🚀 Project Overview
This project builds a **semantic memory system for Machine Learning experiment failures**.
It allows engineers to search past ML failures using natural language and receive actionable insights.

The system uses **Endee as an external high-performance vector database**, accessed via HTTP APIs.
No Docker and no API keys are required.

---

## 🧠 Tech Stack
- Endee (Vector Database)
- Sentence-Transformers (Embeddings)
- Ollama (LLaMA – Local LLM)
- Streamlit (UI)

---

## 🏗️ Architecture
User Query → Embedding → Endee Vector Search → Similar Failures → RAG → Insights

---

## 🧩 Running Endee (Required)

```bash
git clone https://github.com/EndeeLabs/endee.git
cd endee
chmod +x install.sh
./install.sh --release --avx2   # use --neon for Apple Silicon
chmod +x run.sh
./run.sh
```

Verify:
```
http://localhost:8080/api/v1/index/list
```

---

## ⚙️ Run the Project

```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
```

---

## 🔍 Example Queries
- "Model overfits after increasing depth"
- "Training unstable with high learning rate"
- "Loss explodes after few epochs"
