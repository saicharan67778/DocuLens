# ⚡ Queriom | Document Intelligence & Executive Synthesis Engine

Queriom is a production-grade Retrieval-Augmented Generation (RAG) platform that converts unstructured documentation (PDF/TXT) into structured, grounded executive reports and real-time interactive intelligence. 

Powered by Groq's high-speed inference engine, ChromaDB vector storage, FastEmbed dense embeddings, BM25 sparse retrieval, and an integrated Human-in-the-Loop (RLHF) memory system.

---

## 🌟 Key Capabilities

- **Hybrid Document Search:** Merges dense semantic vector search (`BAAI/bge-small-en-v1.5`) with sparse exact-match retrieval (`BM25`) for precision keyword recall.
- **Natural Language Executive Reports:** Takes open-ended directives and generates multi-section briefings, compliance audits, or technical deep-dives with explicit information gap analysis.
- **Semantic Query Memory (RLHF):** Automatically logs questions to SQLite. Human-approved responses (👍) or manual corrections are indexed into a secondary vector store to ground future similar inquiries.
- **DPO Dataset Export:** One-click extraction of prompt-chosen-rejected pairs directly to `.jsonl` for offline Direct Preference Optimization fine-tuning.
- **Grounded Verification:** Includes distance thresholding cutoffs to prevent hallucinations, accompanied by verifiable file name and page citations.
- **Linux Container Hardened:** Pre-configured with SQLite C-extension patching (`pysqlite3-binary`), multi-threaded download locks (`tqdm`), and self-healing vector database initializers for zero-error deployments on Streamlit Cloud.

---

## 🏗️ Technical Architecture

| Layer | Component | Specification |
|---|---|---|
| **Interface** | Streamlit | Custom glassmorphism UI with dual-mode operational tabs |
| **Inference Engine** | Groq LPU | `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-120b` |
| **Embeddings** | FastEmbed | `BAAI/bge-small-en-v1.5` running locally via ONNX Runtime |
| **Vector Database** | ChromaDB | Persistent vector stores for document chunks and RLHF memory |
| **Keyword Search** | Rank-BM25 | In-memory tokenized inverted index |
| **Storage & Logs** | SQLite | Relational database (`interactions.db`) tracking telemetry and edits |
| **Document Parser** | PyPDF | Page-aware text extraction with automatic noise filtration |

---

## 🚀 Local Installation & Execution

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/Queriom.git
cd Queriom

###2. Configure Virtual Environment (Python 3.11 Recommended)

```bash
python -m venv venv

```PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

macOS / Linux:
```Bash
source venv/bin/activate

``` ### 3. Install Dependencies
Bash
pip install --upgrade pip
pip install -r requirements.txt

### 4.Set Environment Variables
Create a .env file in the root directory:

```Bash
GROQ_API_KEY=gsk_your_groq_api_key_here

### 5. Launch Application
```Bash
streamlit run app.py


### Deployment on Streamlit Community Cloud
1.Push your repository to GitHub.

2.Sign in to share.streamlit.io and click Create app.

3.Select your repository, set the branch to main, and set the file path to app.py.

4.Under Settings -> General, set the Python version to 3.11.

5.Under Secrets, configure your Groq key:

Ini, TOML
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"

6.Click Deploy.
