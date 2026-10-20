# 🧠 RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) app built with **LangChain**, **FAISS**, and **Streamlit**.  
It allows you to upload PDFs, split them into chunks, embed them with **HuggingFace sentence-transformers**, store them in FAISS, and answer questions using a **local HuggingFace model** (Flan-T5).  
Everything runs **offline and free** — no OpenAI or HuggingFace API keys required.

---

## 🚀 Features
- 📄 PDF ingestion with `PyPDFLoader`
- ✂️ Text chunking via `RecursiveCharacterTextSplitter`
- 🧩 Free local embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- 📚 Vector storage with FAISS
- 🤖 Local LLM (`google/flan-t5-small`) via HuggingFace Transformers
- 🎨 Streamlit UI for interactive Q&A

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/rag-knowledge-assistant.git
cd rag-knowledge-assistant

pip install -r requirements.txt
