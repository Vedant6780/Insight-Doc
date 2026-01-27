import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Insight AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --primary-hover: #4f46e5;
        --secondary-color: #8b5cf6;
        --background-dark: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --success-color: #10b981;
        --border-color: #334155;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #f1f5f9;
        font-weight: 500;
    }
    
    /* Card-like containers */
    .stExpander {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f1f5f9;
        padding: 12px;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 8px;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Feature cards */
    .feature-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        height: 100%;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    
    .feature-card:hover {
        border-color: #6366f1;
        transform: translateY(-5px);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    h2, h3 {
        color: #f1f5f9;
        font-weight: 600;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: #1e293b;
        border: 1px solid #334155;
        color: #f1f5f9;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: #334155;
        border-color: #6366f1;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #1e293b;
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #6366f1 transparent transparent transparent;
    }
    
    /* Radio button styling */
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.1rem 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
        width: 100%;
        display: block;
        text-align: center;
    }
    
    .stRadio > div > label:hover {
        border-color: #6366f1;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-color: #6366f1;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Divider */
    hr {
        border-color: #334155;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: #1e293b;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# API Keys and Model Configurations
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["👋 Hello! How can I assist you today?"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]

# Part 0: Introduction
def display_introduction():
    # Hero section
    st.markdown("# 🧠 Welcome to Insight AI")
    st.markdown("##### Your Intelligent Document & Text Assistant")
    
    st.markdown("---")
    
    # Features in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📝 Text Processing</h3>
            <p style="color: #94a3b8;">Summarize articles, extract highlights, generate Points of Minutes, and more!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📄 Document Analyzer</h3>
            <p style="color: #94a3b8;">Upload PDFs for comparison, summarization, and intelligent search.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 AI Chat</h3>
            <p style="color: #94a3b8;">Chat with a powerful AI assistant for personalized responses.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Getting started section
    st.markdown("### 🚀 Getting Started")
    st.info("👈 Select a feature from the sidebar to begin. You can also download sample files below to test the features!")
    
    # Sample files with better layout
    st.markdown("### 📥 Sample Files")
    
    sample_files = {
        "📄 Sample Essay": "essay.pdf",
        "📊 Google Report": "google.pdf",
        "🚗 Tesla Report": "tesla.pdf",
        "🚕 Uber Report": "uber.pdf",
        "❓ Sample Questions": "sample_question.pdf"
    }
    
    cols = st.columns(5)
    for idx, (file_name, file_path) in enumerate(sample_files.items()):
        with cols[idx]:
            try:
                st.download_button(
                    label=file_name,
                    data=open(file_path, "rb").read(),
                    file_name=file_path,
                    mime="application/pdf",
                    use_container_width=True
                )
            except FileNotFoundError:
                st.caption(f"{file_name}\n(Not found)")

# Part 1: Text Processing
def process_text_with_groq(task, text):
    try:
        task_map = {
            "📝 Summarize": "Summarize the following text concisely.",
            "✨ Extract Highlights": "Extract the key highlights from the text.",
            "📋 Points of Minutes": "Generate a concise point of minutes (PoM) for the following text.",
            "🎯 Custom Instructions": "Follow the user's custom instructions for the given text."
        }
        user_prompt = f"{task_map[task]}: {text}"
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error accessing Groq API: {e}"

def display_text_processing():
    st.markdown("# 📝 Text Processing")
    st.markdown("##### Transform your text with AI-powered tools")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Settings")
        task = st.radio(
            "Select Task",
            ["📝 Summarize", "✨ Extract Highlights", "📋 Points of Minutes", "🎯 Custom Instructions"],
            index=0,
            help="Choose what you want to do with your text"
        )
        
        st.markdown("---")
        st.markdown("##### 💡 Tips")
        st.caption("• Paste any article or document text")
        st.caption("• Works best with 100-5000 words")
        st.caption("• Custom instructions let you define your own task")
    
    with col2:
        st.markdown("### 📄 Your Text")
        user_input = st.text_area(
            "Paste your text here:",
            key='llama_input',
            height=250,
            placeholder="Enter your article, document, or any text you want to process..."
        )
        
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            submit_button = st.button("🚀 Process", use_container_width=True)
        with col_btn2:
            if user_input:
                st.caption(f"📊 {len(user_input.split())} words")
    
    if submit_button and user_input:
        with st.spinner("🔄 Processing your text..."):
            output = process_text_with_groq(task, user_input)
        
        st.markdown("---")
        st.markdown("### ✅ Result")
        st.success(output)
    elif submit_button and not user_input:
        st.warning("⚠️ Please enter some text to process.")

# Part 2: Document Search
def create_conversational_chain(vector_store):
    llm = HuggingFaceEndpoint(
        repo_id=MODEL_NAME,
        task="text-generation",
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.75,
        top_p=0.9,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    full_answer = result["answer"].strip()

    if "Question:" in full_answer:
        question_start = full_answer.find("Question:")
        cropped_answer = full_answer[question_start:]
    else:
        cropped_answer = "The response does not contain a clearly defined question and answer."

    if "I don't know" in cropped_answer or not cropped_answer.strip():
        cropped_answer = f"The provided context does not provide information on '{query}'."

    history.append((query, cropped_answer))
    return cropped_answer, full_answer, result.get("source_documents", [])

def display_document_search(chain):
    st.markdown("### 🔍 Ask Your Documents")
    
    user_input = st.text_input(
        "Ask a question:",
        key='doc_input',
        placeholder="e.g., What is the revenue for 2023?"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🔎 Search", use_container_width=True)

    if submit_button and user_input:
        with st.spinner("🔄 Searching through documents..."):
            cropped_output, full_output, sources = conversation_chat(user_input, chain, st.session_state['history'])

        st.markdown("---")
        st.markdown("### ✅ Answer")
        st.success(cropped_output)

        with st.expander("📋 View Full Response"):
            st.write(full_output)

        if sources:
            with st.expander("📚 Related Context"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content)
                    st.markdown("---")
    elif submit_button and not user_input:
        st.warning("⚠️ Please enter a question.")

# Part 3: Chat Window
def chat_with_groq(text):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error accessing Groq API: {e}"

def display_chat_window():
    st.markdown("# 💬 Chat with AI")
    st.markdown("##### Have a conversation with our intelligent assistant")
    
    st.markdown("---")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
    
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Your message:",
            key='chat_input',
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.button("📤 Send", use_container_width=True)
    
    if submit_button and user_input:
        with st.spinner("🤔 Thinking..."):
            output = chat_with_groq(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.rerun()
    
    # Clear chat button
    if len(st.session_state['generated']) > 1:
        if st.button("🗑️ Clear Chat"):
            st.session_state['generated'] = ["👋 Hello! How can I assist you today?"]
            st.session_state['past'] = ["Hi!"]
            st.rerun()

# Main Functionality
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 Insight AI")
        st.markdown("---")
        
        st.markdown("### 🧭 Navigation")
        app_mode = st.radio(
            "Select a feature:",
            [
                "🏠 Home",
                "📝 Text Processing", 
                "📄 Document Analyzer", 
                "💬 AI Chat"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Show upload only for Document Analyzer
        if app_mode == "📄 Document Analyzer":
            st.markdown("### 📤 Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                accept_multiple_files=True,
                type=['pdf'],
                help="Upload one or more PDF files to analyze"
            )
        else:
            uploaded_files = None
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
            Made with ❤️ using Streamlit<br>
            Powered by LLaMA & Mistral
        </div>
        """, unsafe_allow_html=True)

    initialize_session_state()
    
    if app_mode == "🏠 Home":
        display_introduction()
    elif app_mode == "📝 Text Processing":
        display_text_processing()
    elif app_mode == "📄 Document Analyzer":
        st.markdown("# 📄 Document Analyzer")
        st.markdown("##### Upload PDFs for intelligent analysis and comparison")
        
        st.markdown("---")
        
        # Instructions
        with st.expander("ℹ️ How to Use", expanded=True):
            st.markdown("""
            **Upload documents using the sidebar**, then ask questions about them!
            
            **Example queries:**
            - 🔄 *"Compare how Tesla and Google incorporate AI into their operations"*
            - 📊 *"What is the total revenue for Google Search in 2023?"*
            - 📝 *"Summarize the key points of the market research"*
            
            **Supported:** PDF files up to 500 pages
            """)
        
        if uploaded_files:
            with st.spinner("📚 Processing your documents..."):
                text = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    file_extension = os.path.splitext(file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(file.read())
                        temp_file_path = temp_file.name

                    if file_extension == ".pdf":
                        loader = PyPDFLoader(temp_file_path)
                        text.extend(loader.load())
                        os.remove(temp_file_path)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                progress_bar.empty()

            st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(text)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
            chain = create_conversational_chain(vector_store)
            
            st.markdown("---")
            display_document_search(chain)
        else:
            st.info("👈 Upload PDF files using the sidebar to get started!")

    elif app_mode == "💬 AI Chat":
        display_chat_window()

if __name__ == "__main__":
    main()
