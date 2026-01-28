# Insight AI Assistant

**Insight AI Assistant** is an interactive Streamlit-based application that offers powerful AI-driven text processing, document analysis (RAG), and conversational capabilities.

## 🚀 Recent Contributions & Updates
This project has been significantly refactored to support modern dependencies and improved stability:
- **LangChain 1.2.7+ Compatibility**: Refactored the codebase to handle the latest LangChain API changes, implementing a robust fallback to `langchain_classic` for preserving legacy chain functionality (`ConversationalRetrievalChain`).
- **Enhanced RAG Stability**: Switched the Document Search engine from the Hugging Face Inference API (which had task compatibility issues) to **Groq (Llama 3)**. This ensures reliable, high-speed document analysis without "text-generation" task rejections.
- **Dependency Clean-up**: Resolved `ModuleNotFoundError` and version conflicts with a clean virtual environment setup.
- **One-Click Launch**: Added `run_app.bat` for easy one-click startup on Windows.

---

## ✨ Features

### 1. **Text Processing**
- **Summarization**: Generate concise summaries of articles or text.
- **Highlights**: Extract key highlights from input text.
- **Points of Minutes (PoM)**: Create structured PoM from provided text.
- **Custom Instructions**: Perform tasks based on user-defined instructions.

### 2. **InsightDoc AI Analyzer (RAG Engine)**
- **PDF Parsing**: Extracts and processes text from PDF documents using `PyPDFLoader`.
- **Vectorization**: Uses local `HuggingFaceEmbeddings` for privacy and speed.
- **Vector Store**: Stores embeddings in `FAISS` for efficient retrieval.
- **Groq Llama 3 Integration**: Uses the powerful **Llama-3.3-70b-versatile** model via Groq for precise, context-aware document Q&A.
- **Usage**: Upload PDFs (e.g., 10-K filings, reports) and ask complex questions like "Compare the revenue of Tesla vs Google".

### 3. **Chat with Assistant**
- Open-ended chat interface powered by Groq (Llama 3).
- Maintains conversation history for context-aware interactions.

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.10 or higher (Tested on 3.13)
- An API Key from [Groq](https://console.groq.com/)
- (Optional) HuggingFace API Key for embeddings

### Setup Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/insight-ai.git
   cd insight-ai
   ```

2. **Set up the environment:**
   We recommend creating a clean virtual environment to avoid conflicts.
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACE_API_KEY=your_hf_api_key_here
   ```

### Running the App
**Option 1: One-Click Script (Windows)**
Run the included batch script:
```powershell
.\run_app.bat
```

**Option 2: Manual Start**
```bash
streamlit run app.py
```

---

## 📂 File Structure
```
.
├── app.py                 # Main application file (Streamlit)
├── requirements.txt       # Project dependencies (LangChain, Streamlit, Groq, FAISS)
├── run_app.bat            # Windows startup script
├── .env                   # API keys (not committed)
├── README.md              # Documentation
└── ...
```

## ⚠️ Troubleshooting
- **ModuleNotFoundError**: Ensure you are running in the correct virtual environment. Use `run_app.bat` to handle this automatically.
- **API Errors**: If you see "Error accessing Groq API", check that your API key is valid in `.env`.
