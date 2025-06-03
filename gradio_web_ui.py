import gradio as gr
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Tuple, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add system paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "personal-rag-system"))
sys.path.append(str(current_dir / "hybrid-rag-system"))

# Import our systems with error handling
PersonalRAGSystem = None
HybridRAGSystem = None

try:
    from rag_multi_docs import PersonalRAGSystem

    logger.info("Personal RAG System imported successfully")
except ImportError as e:
    logger.warning(f"Could not import Personal RAG System: {e}")

try:
    from hybrid_rag import HybridRAGSystem

    logger.info("Hybrid RAG System imported successfully")
except ImportError as e:
    logger.warning(f"Could not import Hybrid RAG System: {e}")

# Web search functionality
from duckduckgo_search import DDGS
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAssistantWebUI:
    def __init__(self):
        """Initialize all AI systems for the web UI"""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)

        # Initialize systems
        self.personal_rag = None
        self.hybrid_rag = None

        try:
            # Import the existing PersonalRAGSystem
            import importlib.util
            rag_file_path = Path(__file__).parent / "personal-rag-system" / "rag_multi-docs.py"
            spec = importlib.util.spec_from_file_location("rag_multi_docs", rag_file_path)
            rag_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rag_module)
            PersonalRAGSystem = rag_module.PersonalRAGSystem

            # Initialize with correct documents folder (just "me" since we're running from project root)
            self.personal_rag = PersonalRAGSystem("me")  # Fixed: don't double the path
            self.personal_rag.load_personal_documents()
            logger.info("Personal RAG System initialized")
        except Exception as e:
            logger.warning(f"Personal RAG System failed to initialize: {e}")

        try:
            # Import the existing HybridRAGSystem
            import importlib.util
            hybrid_file_path = Path(__file__).parent / "hybrid-rag-system" / "hybrid_rag.py"
            spec = importlib.util.spec_from_file_location("hybrid_rag", hybrid_file_path)
            hybrid_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hybrid_module)
            HybridRAGSystem = hybrid_module.HybridRAGSystem

            # Initialize with correct documents folder - use absolute path
            project_root = Path(__file__).parent
            docs_path = project_root / "personal-rag-system" / "me"
            self.hybrid_rag = HybridRAGSystem(str(docs_path), min_context_length=500)
            logger.info("Hybrid RAG System initialized")
        except Exception as e:
            logger.warning(f"Hybrid RAG System failed to initialize: {e}")

    def web_search(self, query: str) -> str:
        """Web search using DuckDuckGo"""
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=3):
                    results.append(f"**{r['title']}**\n{r['href']}\n{r['body']}\n")
            return "\n---\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    def web_search_with_ai(self, query: str) -> Tuple[str, str]:
        """Web search with AI analysis"""
        try:
            # Get web search results
            web_results = self.web_search(query)

            if "Search error" in web_results or "No results found" in web_results:
                return web_results, "❌ Web search failed"

            # Process with AI
            prompt = f"""You are a research assistant. Based on the web search results below, provide a comprehensive answer to the user's question. Always cite your sources with URLs when available.

Web Search Results:
{web_results}

Question: {query}

Answer:"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )

            answer = response.choices[0].message.content
            sources = "🌐 Web Search (DuckDuckGo)"

            return answer, sources

        except Exception as e:
            return f"Error processing web search: {str(e)}", "❌ Error"

    def personal_rag_query(self, query: str) -> Tuple[str, str]:
        """Query personal RAG system"""
        if not self.personal_rag:
            return "Personal RAG System not available. Please ensure documents are in personal-rag-system/me/", "❌ System not available"

        try:
            result = self.personal_rag.query_personal_info(query)
            answer = result.get("answer", "No answer generated")
            sources = f"📚 Personal Documents: {', '.join(result.get('sources', []))}"
            return answer, sources
        except Exception as e:
            return f"Error with Personal RAG: {str(e)}", "❌ Error"

    def hybrid_rag_query(self, query: str) -> Tuple[str, str]:
        """Query hybrid RAG system"""
        if not self.hybrid_rag:
            return "Hybrid RAG System not available. Please check system initialization.", "❌ System not available"

        try:
            answer, metadata = self.hybrid_rag.hybrid_query(query)

            # Format sources based on what was used
            sources_used = metadata.get("sources_used", [])
            if "local" in sources_used and "web" in sources_used:
                sources = f"🔄 Hybrid: Personal docs + Web search"
            elif "local" in sources_used:
                sources = f"🏠 Personal documents only ({metadata.get('local_context_length', 0)} chars)"
            elif "web" in sources_used:
                sources = f"🌐 Web search only"
            else:
                sources = "❓ Unknown sources"

            return answer, sources
        except Exception as e:
            return f"Error with Hybrid RAG: {str(e)}", "❌ Error"

    def compare_all_systems(self, query: str) -> Tuple[str, str, str, str, str, str]:
        """Compare responses from all three systems"""
        # Personal RAG
        personal_answer, personal_sources = self.personal_rag_query(query)

        # Web Search
        web_answer, web_sources = self.web_search_with_ai(query)

        # Hybrid RAG
        hybrid_answer, hybrid_sources = self.hybrid_rag_query(query)

        return personal_answer, personal_sources, web_answer, web_sources, hybrid_answer, hybrid_sources


def create_gradio_interface():
    """Create the Gradio web interface"""

    # Initialize the AI assistant
    try:
        assistant = AIAssistantWebUI()
    except Exception as e:
        logger.error(f"Failed to initialize AI Assistant: {e}")
        assistant = None

    def safe_query(query_func, query: str) -> Tuple[str, str]:
        """Safely execute query with error handling"""
        if not assistant:
            return "AI Assistant not initialized. Please check your setup and API keys.", "❌ System Error"

        if not query.strip():
            return "Please enter a question.", "❌ Empty Query"

        return query_func(query)

    # Define interface functions
    def personal_rag_interface(query: str) -> Tuple[str, str]:
        return safe_query(assistant.personal_rag_query, query)

    def web_search_interface(query: str) -> Tuple[str, str]:
        return safe_query(assistant.web_search_with_ai, query)

    def hybrid_rag_interface(query: str) -> Tuple[str, str]:
        return safe_query(assistant.hybrid_rag_query, query)

    def compare_interface(query: str) -> Tuple[str, str, str, str, str, str]:
        if not assistant:
            error_msg = "AI Assistant not initialized. Please check your setup and API keys."
            return error_msg, "❌ Error", error_msg, "❌ Error", error_msg, "❌ Error"

        if not query.strip():
            empty_msg = "Please enter a question."
            return empty_msg, "❌ Empty", empty_msg, "❌ Empty", empty_msg, "❌ Empty"

        return assistant.compare_all_systems(query)

    # Create the Gradio interface
    with gr.Blocks(
            title="AI Personal Assistant Suite",
            theme=gr.themes.Soft(),
            css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .source-info {
            font-size: 0.9em;
            color: #666;
            font-style: italic;
        }
        """
    ) as demo:

        gr.Markdown("""
        # 🤖 AI Personal Assistant Suite

        Interactive web interface for three complementary AI systems:
        - **Personal RAG**: Query your personal documents
        - **Web Search Agent**: Real-time web research  
        - **Hybrid RAG**: Intelligently combines both sources

        ---
        """)

        with gr.Tabs():
            # Personal RAG Tab
            with gr.TabItem("🏠 Personal RAG System"):
                gr.Markdown("Query your personal documents using advanced RAG technology.")

                with gr.Row():
                    with gr.Column():
                        personal_input = gr.Textbox(
                            label="Ask about your background, experience, or personal information",
                            placeholder="e.g., Tell me about my background and where I'm from",
                            lines=2
                        )
                        personal_btn = gr.Button("Query Personal Documents", variant="primary")

                    with gr.Column():
                        personal_output = gr.Textbox(
                            label="Answer",
                            lines=8,
                            interactive=False
                        )
                        personal_sources = gr.Textbox(
                            label="Sources",
                            lines=1,
                            interactive=False,
                            elem_classes=["source-info"]
                        )

                # Example questions
                gr.Examples(
                    examples=[
                        "Tell me about your background and where you're from",
                        "What are your main technical skills?",
                        "What are your food preferences?",
                        "What is your education background?"
                    ],
                    inputs=personal_input
                )

            # Web Search Tab
            with gr.TabItem("🌐 Web Search Agent"):
                gr.Markdown("Real-time web search with AI-powered analysis.")

                with gr.Row():
                    with gr.Column():
                        web_input = gr.Textbox(
                            label="Search for current information, news, or research topics",
                            placeholder="e.g., Latest developments in AI in 2025",
                            lines=2
                        )
                        web_btn = gr.Button("Search the Web", variant="primary")

                    with gr.Column():
                        web_output = gr.Textbox(
                            label="Answer",
                            lines=8,
                            interactive=False
                        )
                        web_sources = gr.Textbox(
                            label="Sources",
                            lines=1,
                            interactive=False,
                            elem_classes=["source-info"]
                        )

                # Example questions
                gr.Examples(
                    examples=[
                        "Latest developments in AI in 2025",
                        "Current trends in renewable energy",
                        "Who is the CEO of OpenAI?",
                        "Recent breakthroughs in quantum computing"
                    ],
                    inputs=web_input
                )

            # Hybrid RAG Tab
            with gr.TabItem("🔄 Hybrid RAG System"):
                gr.Markdown("Intelligent system that automatically combines personal documents and web search.")

                with gr.Row():
                    with gr.Column():
                        hybrid_input = gr.Textbox(
                            label="Ask any question - the system will choose the best source(s)",
                            placeholder="e.g., How do my skills compare to current AI industry trends?",
                            lines=2
                        )
                        hybrid_btn = gr.Button("Ask Hybrid System", variant="primary")

                    with gr.Column():
                        hybrid_output = gr.Textbox(
                            label="Answer",
                            lines=8,
                            interactive=False
                        )
                        hybrid_sources = gr.Textbox(
                            label="Sources Used",
                            lines=1,
                            interactive=False,
                            elem_classes=["source-info"]
                        )

                # Example questions
                gr.Examples(
                    examples=[
                        "Tell me about your background",
                        "Who is the CEO of OpenAI and what is their background?",
                        "What are the latest AI trends and how do they relate to your skills?",
                        "What are your food preferences and popular restaurants in San Francisco?"
                    ],
                    inputs=hybrid_input
                )

            # Comparison Tab
            with gr.TabItem("⚖️ Compare All Systems"):
                gr.Markdown("Compare responses from all three systems side-by-side.")

                compare_input = gr.Textbox(
                    label="Enter your question to see how each system responds",
                    placeholder="e.g., What are the latest developments in AI?",
                    lines=2
                )
                compare_btn = gr.Button("Compare All Systems", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🏠 Personal RAG")
                        compare_personal = gr.Textbox(label="Answer", lines=6, interactive=False)
                        compare_personal_sources = gr.Textbox(label="Sources", lines=1, interactive=False,
                                                              elem_classes=["source-info"])

                    with gr.Column():
                        gr.Markdown("### 🌐 Web Search")
                        compare_web = gr.Textbox(label="Answer", lines=6, interactive=False)
                        compare_web_sources = gr.Textbox(label="Sources", lines=1, interactive=False,
                                                         elem_classes=["source-info"])

                    with gr.Column():
                        gr.Markdown("### 🔄 Hybrid RAG")
                        compare_hybrid = gr.Textbox(label="Answer", lines=6, interactive=False)
                        compare_hybrid_sources = gr.Textbox(label="Sources", lines=1, interactive=False,
                                                            elem_classes=["source-info"])

                # Example questions for comparison
                gr.Examples(
                    examples=[
                        "Tell me about your background",
                        "Latest developments in AI",
                        "Who is the CEO of OpenAI?",
                        "What are your skills and current job market trends?"
                    ],
                    inputs=compare_input
                )

        # System Status
        with gr.Accordion("🔧 System Status", open=False):
            status_text = []
            if assistant:
                if assistant.personal_rag:
                    status_text.append("✅ Personal RAG System: Ready")
                else:
                    status_text.append("❌ Personal RAG System: Not available")

                if assistant.hybrid_rag:
                    status_text.append("✅ Hybrid RAG System: Ready")
                else:
                    status_text.append("❌ Hybrid RAG System: Not available")

                status_text.append("✅ Web Search Agent: Ready")
            else:
                status_text.append("❌ AI Assistant: Not initialized")

            gr.Markdown("\n".join(status_text))

        # Event handlers
        personal_btn.click(
            fn=personal_rag_interface,
            inputs=personal_input,
            outputs=[personal_output, personal_sources]
        )

        web_btn.click(
            fn=web_search_interface,
            inputs=web_input,
            outputs=[web_output, web_sources]
        )

        hybrid_btn.click(
            fn=hybrid_rag_interface,
            inputs=hybrid_input,
            outputs=[hybrid_output, hybrid_sources]
        )

        compare_btn.click(
            fn=compare_interface,
            inputs=compare_input,
            outputs=[
                compare_personal, compare_personal_sources,
                compare_web, compare_web_sources,
                compare_hybrid, compare_hybrid_sources
            ]
        )

        # Footer
        gr.Markdown("""
        ---
        **AI Personal Assistant Suite** - Powered by OpenAI, ChromaDB, and DuckDuckGo  
        💡 Tip: Try the comparison tab to see how different systems handle the same question!
        """)

    return demo


def main():
    """Launch the Gradio web interface"""
    print("🚀 Starting AI Personal Assistant Web UI...")
    print("📍 Make sure you have:")
    print("   ✅ OpenAI API key in .env file")
    print("   ✅ Personal documents in personal-rag-system/me/")
    print("   ✅ All dependencies installed")
    print()

    try:
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,  # Default Gradio port
            share=False,  # Set to True for public sharing
            show_error=True,  # Show detailed errors
            quiet=False  # Show startup info
        )
    except Exception as e:
        print(f"❌ Failed to launch web UI: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that all dependencies are installed: pip install gradio")
        print("2. Ensure .env file exists with OPENAI_API_KEY")
        print("3. Verify personal documents are in personal-rag-system/me/")


if __name__ == "__main__":
    main()