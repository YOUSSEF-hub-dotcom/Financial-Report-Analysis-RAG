````markdown
# 🚀 Financial Report Analysis RAG

> 🏆 This repository is my official submission for the **Tips Hindawi Challenge (June–July) 2026**.

## 👤 Participant

| Field | Value |
|-------|-------|
| **Full Name** | Youssef Mahmoud AboAli |
| **Project Name** | Financial Report Analysis RAG |
| **GitHub Username** | YOUSSEF-hub-dotcom |
| **Challenge Batch** | June–July 2026 |
| **Training Program** | Large Language Models (LLMs) Program |
| **Organization** | Edrak for AI |

---

# 📖 Project Overview

Financial Report Analysis RAG is an end-to-end Retrieval-Augmented Generation (RAG) system designed to analyze financial reports and answer user questions using relevant document context.

The project consists of two main phases:

- **Offline Ingestion:** Parsing financial documents, extracting metadata, chunking content, generating embeddings, and indexing data into MongoDB and Qdrant.
- **Online Retrieval & Generation:** Retrieving relevant document chunks, generating answers using Groq LLMs, validating responses through guardrails, and returning structured JSON outputs.

The system also provides a FastAPI backend, Streamlit dashboard, Redis caching, and MLflow experiment tracking.

---

# ✨ Features

- Financial document ingestion and preprocessing.
- HTML table parsing and Markdown conversion.
- Metadata extraction for financial documents.
- Hybrid 3-tier chunking strategy.
- Dual storage architecture using MongoDB and Qdrant.
- Vector search with metadata filtering.
- Response generation using Groq Large Language Models.
- Automatic model fallback mechanism.
- Async guardrail validation for numerical verification.
- Redis query caching.
- REST API using FastAPI.
- Interactive Streamlit dashboard.
- MLflow experiment tracking and structured logging.

---

# 🛠️ Technologies Used

### Programming Language

- Python 3.11+

### Data Processing

- BeautifulSoup4
- lxml
- Pandas
- Tabulate
- tiktoken

### AI & LLM

- LangChain
- LangChain-Groq
- Groq
- llama-3.3-70b-versatile
- qwen/qwen3.6-27b
- nomic-ai/nomic-embed-text-v1.5

### Databases

- MongoDB
- Qdrant
- Redis

### Backend

- FastAPI
- Uvicorn

### Dashboard

- Streamlit

### Monitoring

- MLflow
- Structured JSON Logging

### Configuration

- python-dotenv

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Financial_RAG
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and configure the required environment variables before running the project.

Run the FastAPI server:

```bash
uvicorn app.api.main:app --reload
```

Run the Streamlit dashboard:

```bash
streamlit run app/ui/streamlit_app.py
```

---

# 🚀 Usage

1. Prepare a supported financial document.
2. Run the ingestion pipeline to parse and index the document.
3. Store document metadata in MongoDB and vector embeddings in Qdrant.
4. Launch the FastAPI service or Streamlit dashboard.
5. Submit financial questions.
6. Retrieve the most relevant document context.
7. Generate validated answers through the RAG pipeline.

---

# 📸 Demo

You can include:

- Streamlit Dashboard
- FastAPI Swagger Interface
- Document Upload Page
- Chat Interface
- Sample Financial Question & Answer

---

# 📈 Results

### Project Status

- ✅ Offline Ingestion Pipeline completed.
- ✅ Generation Engine completed.
- ✅ Pipeline Orchestrator completed.
- ✅ FastAPI Backend completed.
- ✅ Multi-format Upload Parser completed.
- ✅ Streamlit Dashboard completed.

### Testing

- **148 / 148 Unit Tests Passed**

The project is fully implemented and ready for deployment.

---

# 🔮 Future Improvements

- Deployment of the complete system.
- Production infrastructure setup.
- Continuous monitoring and maintenance.

---

# 📚 About the Challenge

This project was developed as part of the **Tips Hindawi Challenge (June–July 2026)** under the **Large Language Models (LLMs) Program**.

The challenge encourages participants to build practical AI solutions, apply software engineering best practices, and publish production-oriented projects through GitHub.

---

# 📄 License

This project is shared for educational and portfolio purposes.
````
