import streamlit as st
import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai
import json

HISTORY_FILE = "chat_history.json"

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(messages):
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f)

# Load environment variables (override to ensure new keys are picked up without restarting)
load_dotenv(override=True)

if "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Google API Key not found! Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# --- Page Config & Styling ---
st.set_page_config(page_title="FeePal Chatbot", page_icon="💡", layout="centered")

# Custom CSS for White, Blue, Teal scheme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Main background */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 15px rgba(0,0,0,0.03);
    }
    
    /* Headers and Text */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    p, span {
        color: #334155 !important;
    }
    
    /* Gradient Title */
    .premium-title {
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: #ffffff;
        color: #0f172a !important;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button p {
        color: #0f172a !important;
    }
    .stButton>button:hover {
        background: #f1f5f9;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-color: #94a3b8;
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: none;
    }
    
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
    }
    [data-testid="stChatMessage"] p {
        color: #1e293b !important;
        line-height: 1.6;
    }
    
    /* User chat bubble */
    [data-testid="stChatMessage"]:nth-child(even) {
        background: linear-gradient(to bottom right, #f0fdfa, #ffffff);
        border: 1px solid #ccfbf1;
    }
    
    /* Chat Input */
    [data-testid="stChatInput"] {
        border-radius: 24px;
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        background-color: white;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #0ea5e9;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15);
    }
    
    /* Stop generation button */
    .stop-btn-wrapper {
        position: fixed;
        bottom: 90px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
    }
    .stop-btn-wrapper div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #ef4444, #b91c1c) !important;
        color: white !important;
        border-radius: 20px;
        padding: 5px 20px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.4);
        width: auto !important;
        display: inline-block;
    }
    .stop-btn-wrapper div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #dc2626, #991b1b) !important;
        box-shadow: 0 6px 8px -1px rgba(239, 68, 68, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# --- Caching RAG Initialization ---
@st.cache_resource(show_spinner=False)
def initialize_models():
    """Initializes and caches the LLM and Embedding models."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
        
    llm = Gemini(model="models/gemini-2.5-flash", temperature=0.8, api_key=api_key)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50
    return llm, embed_model

@st.cache_resource(show_spinner=False)
def initialize_vector_store():
    """Initializes and caches the ChromaDB vector store connection."""
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("feepal_report")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return chroma_collection, vector_store, storage_context

@st.cache_resource(show_spinner=False)
def get_chat_engine(_vector_store):
    """Loads the chat engine from the vector store."""
    try:
        index = VectorStoreIndex.from_vector_store(_vector_store)
        return index.as_chat_engine(chat_mode="condense_plus_context", similarity_top_k=3)
    except Exception as e:
        return None


# --- App Logic & UI ---

st.markdown('<div class="premium-title">💡 FeePal Assistant</div>', unsafe_allow_html=True)
st.markdown("Welcome to FeePal Chatbot. Ask me anything about the FeePal architecture, features, and technical stack!")

# 1. Init Models & DB
with st.spinner("Initializing AI Models..."):
    initialize_models()
    chroma_collection, vector_store, storage_context = initialize_vector_store()

# Sidebar for Ingestion
with st.sidebar:
    st.header("⚙️ Administration")
    
    if st.button("＋ New Chat"):
        if "messages" in st.session_state:
            st.session_state.messages = []
            save_chat_history([])
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()
            
    if st.button("⟳ Reload Knowledge Base"):
        with st.spinner("Ingesting documents..."):
            try:
                data_dir = "./data"
                if not os.path.exists(data_dir):
                    st.error(f"Directory {data_dir} not found.")
                else:
                    documents = SimpleDirectoryReader(data_dir).load_data()
                    if documents:
                        index = VectorStoreIndex.from_documents(
                            documents, storage_context=storage_context, show_progress=False
                        )
                        st.success(f"Ingested {len(documents)} document parts!")
                        # Force clear cache for chat engine to reload
                        st.cache_resource.clear()
                    else:
                        st.warning("No documents found in data directory.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                
    if st.button("✕ Delete Chat History"):
        if "messages" in st.session_state:
            st.session_state.messages = []
            save_chat_history([])
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()
            
# 2. Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Try to get or initialize chat engine
if "chat_engine" not in st.session_state:
    engine = get_chat_engine(vector_store)
    if engine is None:
        st.warning("⚠️ Chat engine not initialized. Please click 'Reload Knowledge Base' in the sidebar.")
    else:
        st.session_state.chat_engine = engine

# 3. Display Chat Messages
for message in st.session_state.messages:
    # Basic styling to distinguish roles if needed, though CSS handles mostly
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input & Processing
if prompt := st.chat_input("Ask about FeePal..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if "chat_engine" in st.session_state and st.session_state.chat_engine is not None:
        with st.chat_message("assistant"):
            stop_placeholder = st.empty()
            with stop_placeholder.container():
                st.markdown('<div class="stop-btn-wrapper">', unsafe_allow_html=True)
                st.button("🛑 Stop", key="stop_gen_btn")
                st.markdown('</div>', unsafe_allow_html=True)

            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    st.markdown(str(response))
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    save_chat_history(st.session_state.messages)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
            
            stop_placeholder.empty()
    else:
        with st.chat_message("assistant"):
            st.warning("Chat engine is not ready yet.")

# Update the Sidebar Chat History at the very end so it shows the latest messages
with st.sidebar:
    st.divider()
    st.header("💬 Chat History")
    
    user_messages = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if user_messages:
        chat_title = user_messages[0]
        if len(chat_title) > 35:
            chat_title = chat_title[:32] + "..."
        st.info(f"**Current Session:**\n{chat_title}")
    else:
        st.markdown("*No active chat yet.*")
