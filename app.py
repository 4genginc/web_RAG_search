import gradio as gr
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import traceback

# Setup path for hybrid rag import
project_dir = Path(__file__).parent.resolve()
hybrid_path = project_dir / "hybrid-rag-system"
sys.path.append(str(hybrid_path))

# Load env vars (for OpenAI etc)
load_dotenv()

# Import HybridRAGSystem class
try:
    import importlib.util
    hrag_py = hybrid_path / "hybrid_rag.py"
    spec = importlib.util.spec_from_file_location("hybrid_rag", hrag_py)
    hrag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hrag_module)
    HybridRAG = getattr(hrag_module, "HybridRAGSystem")
except Exception as e:
    print(f"[ERROR] Failed to import Hybrid RAG: {e}\n{traceback.format_exc()}")
    HybridRAG = None

# Initialize hybrid system
hybrid_rag = None
try:
    docs_path = project_dir / "personal-rag-system" / "me"
    hybrid_rag = HybridRAG(str(docs_path), min_context_length=500)
except Exception as e:
    print(f"[ERROR] Failed to initialize HybridRAG: {e}\n{traceback.format_exc()}")

# Define chat function for Gradio
def chat_fn(message, history):
    if not hybrid_rag:
        return "❌ Hybrid RAG system not available. Please check backend logs."
    try:
        answer, metadata = hybrid_rag.hybrid_query(message)
        return answer
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"

# Launch super-simple Gradio chat
gr.ChatInterface(
    fn=chat_fn,
    chatbot=gr.Chatbot(show_label=False, show_copy_button=True),
    textbox=gr.Textbox(placeholder="Ask anything...", show_label=False, container=False),
    title=None,
    description=None,
    theme=gr.themes.Default(),
).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
