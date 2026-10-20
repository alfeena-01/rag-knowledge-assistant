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

## 📋 Requirements
Here’s the recommended requirements.txt:

langchain==0.2.11
langchain-community==0.2.4
langchain-text-splitters==0.2.4
langchain-openai==0.1.7   # optional, not used here
langsmith==0.1.147
faiss-cpu==1.7.4
streamlit==1.32.0
PyPDF2==3.0.1
pypdf==4.0.1
sentence-transformers
transformers
torch
python-dotenv


## 🛠️ Usage
Run the app:

streamlit run app.py


Steps:

Upload a PDF file in the Streamlit UI.

The assistant will:

Load and split the PDF into chunks.

Embed chunks with HuggingFace embeddings.

Store them in FAISS.

Retrieve relevant chunks.

Generate answers using a local HuggingFace model.

## ⚡ Notes
✅ No API keys required — everything runs locally.

🔄 You can swap flan-t5-small for larger models (flan-t5-base, flan-t5-large) if your machine can handle it.

🛡️ If you later want to use API-based models (OpenAI or HuggingFaceHub), add your keys to a .env file and keep it out of GitHub with .gitignore.

## 📂 Project Structure

rag-knowledge-assistant/
│── app.py              # Streamlit entry point
│── rag_pipeline.py     # RAG pipeline logic
│── data/               # Folder for uploaded PDFs
│── requirements.txt    # Dependencies
│── README.md           # Project documentation


## 🎯 Next Steps
Try larger HuggingFace models for better answers.

Add support for multiple PDFs.

Extend to other file types (Word, text, etc.).

Experiment with GPT4All or other local LLMs for offline Q&A.

## 🙌 Contributing
Pull requests are welcome!
If you’d like to add new features (multi-file support, UI improvements, new models), feel free to fork and submit a PR.

## 📜 License
This project is open-source under the MIT License.
You are free to use, modify, and distribute it with attribution.