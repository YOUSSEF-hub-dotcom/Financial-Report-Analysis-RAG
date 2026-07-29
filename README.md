# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026

> 🏆 This repository is my official submission for the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Youssef Mahmoud                      |
| Project Name     | Financial RAG — AI SEC Filing & Report Analyzer |
| GitHub Username  | [https://github.com/YOUSSEF-hub-dotcom]               |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

**Financial RAG** is an enterprise-grade Retrieval-Augmented Generation (RAG) system engineered specifically to parse, chunk, index, and analyze complex SEC Filings (10-K, 10-Q) and financial statements for major tech companies (e.g., Apple, Microsoft, NVIDIA).

The pipeline resolves common RAG challenges such as table/text disambiguation, semantic loss during chunking, and latency during peak usage by leveraging dual-database storage (Qdrant Vector DB + MongoDB Document Store) alongside Redis caching and robust async API warm-up mechanisms.

---

# ✨ Features

* **Multi-Format Ingestion Engine**: Automatically parses complex HTML financial tables and text with structure preservation.
* **Hybrid Storage Architecture**: Store dense vectors in **Qdrant** for high-precision similarity search, while anchoring source metadata & original text in **MongoDB**.
* **Redis Caching & Latency Optimization**: Instant retrieval for frequent financial queries using Redis semantic caching.
* **System Health & API Warm-up**: Self-healing API endpoints with async model warm-up indicators (`healthy` / `degraded`) built into the Streamlit UI.
* **Interactive Streamlit Dashboard**: Clean financial chat interface supporting real-time query answers, document context display, and upload capabilities.

---

# 🛠️ Technologies Used

* **Language & Frameworks**: Python 3.12, FastAPI, Streamlit, PyTorch (CUDA acceleration)
* **Vector Database**: Qdrant
* **Document & Cache Storage**: MongoDB, Redis
* **LLM & Embeddings Engine**: HuggingFace Transformers, Sentence-Transformers, OpenAI API
* **Evaluation & Experiment Tracking**: MLflow, Pytest
* **Data Processing**: PyPDF, python-docx, html5lib, PyPDF2

---

# ⚙️ Installation

### 1. Clone the Repository & Setup Virtual Environment
```bash
git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/[YOUR_REPO_NAME].git
cd Financial_RAG

python3.12 -m venv Financial_env
source Financial_env/bin/activate
pip install -r requirements.txt
2. Environment Variables Setup
Create a .env file in the root directory:

مقتطف الرمز
OPENAI_API_KEY=your_openai_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
MONGO_URI=mongodb://localhost:27017
REDIS_HOST=localhost
REDIS_PORT=6379
3. Ensure Local Services are Running
Bash
# Start MongoDB & Redis
sudo service redis-server start
sudo mongod --dbpath /var/lib/mongodb --fork --logpath /var/log/mongodb/mongod.log
🚀 Usage
Step 1: Run Ingestion Pipeline
To parse SEC HTML filings and populate vector embeddings into Qdrant & MongoDB:

Bash
python src/1_ingestion/run_ingestion.py
Step 2: Start the FastAPI Backend
Bash
uvicorn app.api.main:app --reload --port 8000
Step 3: Launch the Streamlit Frontend
Bash
streamlit run app/ui/streamlit_app.py
Open http://localhost:8501 in your browser to interact with the system.

📸 Demo
(Add screenshots or GIFs of your Streamlit UI showing the Green Health Status and Chat Answers here)

📈 Results
Successfully ingested over 700+ high-density chunks across SEC financial reports with zero structure loss.

Reduced first-query latency via async backend warm-up routines.

Achieved seamless 1-to-1 chunk mapping between Qdrant vector IDs and MongoDB raw document logs.

🔮 Future Improvements
Integrate GraphRAG (Knowledge Graphs) for entity-relationship tracking across financial quarters.

Implement cross-encoder reranking (e.g., Cohere/BGE Reranker) for top-k retrieval enhancement.

Add native multi-modal support for financial charts and infographic PDF extractions.

📚 About the Challenge
This project was developed as part of the Tips Hindawi Challenge (June–July) 2026.

Tips Hindawi is the internships department of Edrak for Ai, and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official Tips Hindawi website.

📄 License
This project is shared for educational and portfolio purposes.