import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import logging
import re
from typing import List, Dict, Any, Optional

# PDF processing imports
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonalRAGSystem:
    def __init__(self, documents_folder: str = "me"):
        """Initialize the Personal RAG System
        
        Args:
            documents_folder: Path to folder containing personal documents (default: "me")
        """
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        # Create path to documents folder relative to script location
        self.documents_folder = script_dir / documents_folder
        self._validate_setup()
        
        # Load API key
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your OpenAI API key.")
        
        self.client = OpenAI(api_key=self.api_key)
        self._setup_vector_store()
    
    def _validate_setup(self):
        """Validate project setup and dependencies"""
        # Check if documents folder exists
        if not self.documents_folder.exists():
            script_location = Path(__file__).parent.absolute()
            logger.error(f"Documents folder '{self.documents_folder}' not found!")
            logger.info(f"Please create the 'me/' folder in the same directory as this script: {script_location}")
            logger.info("Expected structure:")
            logger.info(f"{script_location.name}/")
            logger.info("├── rag_multi-docs.py")
            logger.info("├── me/")
            logger.info("│    ├── summary.txt")
            logger.info("│    └── resume.pdf")
            logger.info("└── ...")
            logger.info(f"\nCreate the folder with: mkdir {self.documents_folder}")
            sys.exit(1)
        
        # Check for required files
        summary_file = self.documents_folder / "summary.txt"
        resume_file = self.documents_folder / "resume.pdf"
        
        if not summary_file.exists():
            logger.warning(f"Summary file not found: {summary_file}")
            logger.info("Create me/summary.txt with your personal introduction")
        
        if not resume_file.exists():
            logger.warning(f"Resume file not found: {resume_file}")
            logger.info("Add me/resume.pdf with your professional resume")
        
        if not summary_file.exists() and not resume_file.exists():
            logger.error("No personal documents found! Please add summary.txt and/or resume.pdf to the me/ folder.")
            sys.exit(1)
        
        # Check PDF processing capabilities
        if resume_file.exists() and not any([PyPDF2, pdfplumber, PdfReader]):
            logger.error("PDF file found but no PDF processing library available!")
            logger.info("Install PDF processing libraries: pip install PyPDF2 pdfplumber pypdf")
            sys.exit(1)
    
    def _setup_vector_store(self):
        """Initialize ChromaDB with OpenAI embeddings"""
        try:
            self.chroma_client = chromadb.Client()
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name="text-embedding-3-small"
            )
            
            # Create collection for personal documents
            collection_name = "personal_profile"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    collection_name, 
                    embedding_function=self.embed_fn
                )
                logger.info(f"Created new collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup vector store: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using available libraries"""
        text = ""
        
        # Try pdfplumber first (best for complex layouts)
        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                logger.info(f"Successfully extracted text using pdfplumber: {len(text)} characters")
                return text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        # Try pypdf
        if PdfReader:
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                logger.info(f"Successfully extracted text using pypdf: {len(text)} characters")
                return text
            except Exception as e:
                logger.warning(f"pypdf failed: {e}")
        
        # Try PyPDF2 as fallback
        if PyPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                logger.info(f"Successfully extracted text using PyPDF2: {len(text)} characters")
                return text
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")
        
        logger.error("All PDF extraction methods failed!")
        return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        if not text.strip():
            return []
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def load_personal_documents(self):
        """Load and process personal documents from the me/ folder"""
        documents = []
        metadatas = []
        ids = []
        doc_count = 0
        
        # Load summary.txt
        summary_file = self.documents_folder / "summary.txt"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_text = f.read().strip()
                
                if summary_text:
                    documents.append(summary_text)
                    metadatas.append({
                        "source": "summary",
                        "type": "personal_intro",
                        "category": "background",
                        "file": "summary.txt"
                    })
                    ids.append("summary_1")
                    doc_count += 1
                    logger.info(f"Loaded summary.txt: {len(summary_text)} characters")
                
            except Exception as e:
                logger.error(f"Failed to load summary.txt: {e}")
        
        # Load and process resume.pdf or linkedin.pdf
        resume_file = self.documents_folder / "resume.pdf"
        linkedin_file = self.documents_folder / "linkedin.pdf"
        
        pdf_file = None
        if resume_file.exists():
            pdf_file = resume_file
            logger.info(f"Found resume.pdf")
        elif linkedin_file.exists():
            pdf_file = linkedin_file
            logger.info(f"Found linkedin.pdf")
        
        if pdf_file:
            try:
                resume_text = self.extract_text_from_pdf(pdf_file)
                
                if resume_text.strip():
                    # Chunk the resume for better retrieval
                    resume_chunks = self.chunk_text(resume_text, chunk_size=600, overlap=100)
                    
                    for i, chunk in enumerate(resume_chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "source": "resume",
                            "type": "professional_info",
                            "category": f"resume_chunk_{i+1}",
                            "file": pdf_file.name
                        })
                        ids.append(f"resume_chunk_{i+1}")
                        doc_count += 1
                    
                    logger.info(f"Loaded {pdf_file.name}: {len(resume_chunks)} chunks from {len(resume_text)} characters")
                else:
                    logger.warning(f"{pdf_file.name} appears to be empty or unreadable")
                
            except Exception as e:
                logger.error(f"Failed to load {pdf_file.name}: {e}")
        else:
            logger.warning("No resume.pdf or linkedin.pdf found")
        
        # Add documents to vector store
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully loaded {doc_count} documents into vector store")
            except Exception as e:
                logger.error(f"Failed to add documents to vector store: {e}")
                raise
        else:
            logger.warning("No documents were loaded! Please check your files in the 'me/' folder.")
    
    def query_personal_info(self, query: str, n_results: int = 3, max_tokens: int = 500) -> Dict[str, Any]:
        """Query personal information using RAG"""
        try:
            # Retrieve relevant documents
            results = self.collection.query(
                query_texts=[query], 
                n_results=n_results
            )
            
            if not results["documents"][0]:
                return {
                    "answer": "I don't have information about that topic in my knowledge base.",
                    "sources": [],
                    "retrieved_docs": []
                }
            
            # Prepare context from retrieved documents
            context_docs = results["documents"][0]
            context_sources = [meta.get("source", "unknown") for meta in results["metadatas"][0]]
            context_files = [meta.get("file", "unknown") for meta in results["metadatas"][0]]
            
            # Create enriched context
            context_parts = []
            for doc, source, file in zip(context_docs, context_sources, context_files):
                context_parts.append(f"[{source.upper()} from {file}]: {doc}")
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using LLM
            prompt = self._create_personal_prompt(context, query)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            logger.info(f"Retrieved {len(context_docs)} documents for query: '{query[:50]}...'")
            
            return {
                "answer": answer,
                "sources": list(set(context_sources)),
                "files": list(set(context_files)),
                "retrieved_docs": context_docs
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "retrieved_docs": []
            }

    def _create_personal_prompt(self, context: str, query: str) -> str:
        """Create a personalized prompt with strict constraints to prevent hallucination"""
        return f"""You are Rongjun Geng answering questions about yourself using ONLY your personal documents.

    Context Information from Personal Documents:
    {context}

    STRICT INSTRUCTIONS:
    - Answer using ONLY the information provided in the context above
    - DO NOT add information from your general training knowledge
    - DO NOT speculate about current market trends, industry conditions, or external information
    - If asked about external information (market trends, current events, other people), respond: "I don't have that information in my personal documents. I can only answer questions about my background, skills, and experience."
    - Always respond in first person as Rongjun Geng ("I am originally from..." not "The person is from...")
    - Be specific about your background, skills, work experience, and personal details when available in the context
    - If the context doesn't contain enough information about yourself, say so honestly

    Question: {query}

    Answer as Rongjun Geng using ONLY the personal document information above:"""

def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)

def print_query_result(query: str, result: Dict[str, Any]):
    """Pretty print query results"""
    print(f"🤔 Question: {query}")
    print("─" * 60)
    print(f"💬 Answer: {result['answer']}")
    
    if result.get('sources'):
        print(f"📚 Sources: {', '.join(result['sources'])}")
    
    if result.get('files'):
        print(f"📄 Files: {', '.join(result['files'])}")
    
    print()
    print_separator()
    print()

def main():
    """Main function to demonstrate the Personal RAG System"""
    print_separator()
    print("🚀 Personal RAG System")
    print_separator()
    print()
    
    try:
        # Initialize the system
        print("📂 Initializing Personal RAG System...")
        rag_system = PersonalRAGSystem()
        
        # Load personal documents
        print("📖 Loading personal documents...")
        rag_system.load_personal_documents()
        print("✅ Documents loaded successfully!")
        print()
        
        # Example queries to demonstrate the system
        sample_queries = [
            "Tell me about your background and where you're from",  # Better: "your" instead of "my"
            "What are your main technical skills and expertise?",    # Better: professional perspective
            "What is your current job and responsibilities?",        # Natural conversation style
            "What companies have you worked for?",                   # Direct and professional
            "What is your education background?",                    # Clear and natural
            "What experience do you have with AI and machine learning?",  # Specific technical question
            "Tell me about your leadership experience",              # Professional inquiry
            "What certifications or achievements do you have?",      # Portfolio-style question
            "What are your personal interests or hobbies?"          # Getting to know you question
        ]
        
        print("🎯 Running Sample Queries...")
        print_separator()
        print()
        
        # Process sample queries
        for query in sample_queries:
            result = rag_system.query_personal_info(query)
            print_query_result(query, result)
        
        # Interactive mode
        print("🎮 Interactive Mode - Ask anything!")
        print("(Type 'quit', 'exit', or 'q' to exit)")
        print()
        
        while True:
            try:
                user_query = input("Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q', '']:
                    print("\n👋 Thanks for using the Personal RAG System!")
                    break
                
                print("\n🔍 Processing your question...\n")
                result = rag_system.query_personal_info(user_query)
                
                print(f"💬 Answer: {result['answer']}")
                if result.get('sources'):
                    print(f"📚 Sources: {', '.join(result['sources'])}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Personal RAG System!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print()
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have created a .env file with your OPENAI_API_KEY")
        print("2. Ensure the 'me/' folder exists with summary.txt and resume.pdf")
        print("3. Check that all required packages are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()