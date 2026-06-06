import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Initialize Gemini LLM and local Embedding
# Temperature set to 0.8 for varied responses
llm = Gemini(model="models/gemini-2.5-flash", temperature=0.8)
# Using a local embedding model to avoid Gemini API free-tier rate limits
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 500
Settings.chunk_overlap = 50

# ChromaDB Setup
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("feepal_report")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

_chat_engine = None

def ingest_document():
    """Reads the files from data/ directory and ingests it into ChromaDB if changed."""
    global _chat_engine
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found. Please ensure data directory exists.")
        
    # Simple optimization: if we already have documents in DB, skip re-ingestion
    # Unless forced, we rely on the persistent DB.
    if chroma_collection.count() > 0:
        if _chat_engine is None:
            get_chat_engine()
        return {"status": "success", "message": "Database already loaded. Ready to chat!"}

    documents = SimpleDirectoryReader(data_dir).load_data()
    
    # Check if there are documents
    if not documents:
        raise ValueError("No documents found in data directory.")
        
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    
    # Initialize chat engine with conversation memory and top 3 retrieval
    _chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", similarity_top_k=3)
    return {"status": "success", "message": f"Ingested {len(documents)} document parts."}

def get_chat_engine():
    global _chat_engine
    if _chat_engine is not None:
        return _chat_engine
        
    # If not in memory, try to load from existing index
    try:
        if chroma_collection.count() > 0:
            index = VectorStoreIndex.from_vector_store(
                vector_store,
            )
            _chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", similarity_top_k=3)
            return _chat_engine
    except Exception as e:
        print(f"Failed to load existing index: {e}")
        
    return None

def chat(message: str):
    engine = get_chat_engine()
    if not engine:
        raise ValueError("Chat engine not initialized. Please call /ingest first.")
        
    response = engine.chat(message)
    return str(response)

def reset_chat():
    global _chat_engine
    engine = get_chat_engine()
    if engine:
        engine.reset()
