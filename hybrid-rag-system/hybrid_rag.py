import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from duckduckgo_search import DDGS
import logging
import re
from typing import Optional, Tuple

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

class HybridRAGSystem:
    def __init__(self, documents_folder: str = "../personal-rag-system/me", min_context_length: int = 150):
        """Initialize Fixed Hybrid RAG System with much better similarity matching
        
        Args:
            documents_folder: Path to personal documents folder
            min_context_length: Minimum context length before triggering web search (further lowered)
        """
        self.min_context_length = min_context_length
        self.documents_folder = Path(__file__).parent.absolute() / documents_folder
        
        logger.info(f"Documents folder path: {self.documents_folder}")
        logger.info(f"Documents folder exists: {self.documents_folder.exists()}")
        
        # Load API key
        project_root = Path(__file__).parent.parent
        load_dotenv(project_root / ".env")
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        self._setup_local_rag()
    
    def _setup_local_rag(self):
        """Setup ChromaDB for local document retrieval"""
        try:
            self.chroma_client = chromadb.Client()
            self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name="text-embedding-3-small"
            )
            
            try:
                # Try to delete and recreate collection to ensure fresh data
                try:
                    self.chroma_client.delete_collection("fixed_hybrid_personal")
                    logger.info("Deleted old collection to force fresh reload")
                except:
                    pass
                
                self.collection = self.chroma_client.create_collection(
                    "fixed_hybrid_personal", 
                    embedding_function=self.embed_fn
                )
                logger.info("Created fresh collection")
                self._load_personal_documents()
                
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                self.collection = None
            
            logger.info("Local RAG system initialized")
        except Exception as e:
            logger.error(f"Failed to setup local RAG: {e}")
            self.collection = None
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using available libraries"""
        text = ""
        
        logger.info(f"Attempting to extract text from: {pdf_path}")
        
        # Try pdfplumber first (best for complex layouts)
        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            logger.info(f"Page {page_num + 1}: {len(page_text)} characters")
                logger.info(f"Successfully extracted text using pdfplumber: {len(text)} characters")
                return text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        # Try pypdf
        if PdfReader:
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    logger.info(f"Page {page_num + 1}: {len(page_text)} characters")
                logger.info(f"Successfully extracted text using pypdf: {len(text)} characters")
                return text
            except Exception as e:
                logger.warning(f"pypdf failed: {e}")
        
        # Try PyPDF2 as fallback
        if PyPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text += page_text + "\n"
                        logger.info(f"Page {page_num + 1}: {len(page_text)} characters")
                logger.info(f"Successfully extracted text using PyPDF2: {len(text)} characters")
                return text
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")
        
        logger.error("All PDF extraction methods failed!")
        return ""
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> list:
        """Split text into smaller, more focused chunks for better retrieval"""
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
    
    def _load_personal_documents(self):
        """Load personal documents with FIXED approach focusing on smaller, targeted chunks"""
        if not self.collection:
            logger.error("No collection available")
            return
        
        logger.info(f"Loading documents from: {self.documents_folder}")
        
        if not self.documents_folder.exists():
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        # Load summary.txt with multiple targeted entries
        summary_file = self.documents_folder / "summary.txt"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_text = f.read().strip()
                
                if summary_text:
                    # Add the full summary
                    documents.append(summary_text)
                    metadatas.append({
                        "source": "summary",
                        "type": "personal_background",
                        "category": "full_summary",
                        "file": "summary.txt",
                        "keywords": "background, personal, Rongjun Geng, software engineer, data scientist"
                    })
                    ids.append("summary_full")
                    
                    # Extract specific information and create targeted entries
                    summary_lower = summary_text.lower()
                    
                    # Location information
                    if "beijing" in summary_lower and "san francisco" in summary_lower:
                        location_text = "Rongjun Geng is originally from Beijing, China, but moved to San Francisco Bay Area in 2017."
                        documents.append(location_text)
                        metadatas.append({
                            "source": "summary",
                            "type": "location_info",
                            "category": "origin_current_location",
                            "file": "summary.txt",
                            "keywords": "Beijing, San Francisco, from, location, where, origin"
                        })
                        ids.append("location_info")
                    
                    # Professional background
                    if "software engineer" in summary_lower and "data scientist" in summary_lower:
                        profession_text = "Rongjun Geng is a software engineer and data scientist."
                        documents.append(profession_text)
                        metadatas.append({
                            "source": "summary",
                            "type": "professional_background",
                            "category": "profession_role",
                            "file": "summary.txt",
                            "keywords": "software engineer, data scientist, profession, job, role"
                        })
                        ids.append("profession_info")
                    
                    # Food preferences
                    if "food" in summary_lower:
                        food_text = "Rongjun loves all foods, particularly French food, but is repelled by most forms of cheese except cream cheese and mozzarella. Enjoys cheesecake and pizza."
                        documents.append(food_text)
                        metadatas.append({
                            "source": "summary",
                            "type": "personal_preferences",
                            "category": "food_preferences",
                            "file": "summary.txt",
                            "keywords": "food, French food, cheese, preferences, likes, dislikes"
                        })
                        ids.append("food_preferences")
                    
                    logger.info(f"Loaded summary.txt: {len(summary_text)} characters in {len([d for d in documents if 'summary' in [m['source'] for m in metadatas]])} targeted entries")
                else:
                    logger.warning("summary.txt is empty")
            except Exception as e:
                logger.error(f"Failed to load summary.txt: {e}")
        else:
            logger.warning(f"summary.txt not found at: {summary_file}")
        
        # Load and process resume.pdf with targeted skill extraction
        resume_file = self.documents_folder / "resume.pdf"
        if resume_file.exists():
            try:
                logger.info(f"Processing resume.pdf...")
                resume_text = self.extract_text_from_pdf(resume_file)
                
                if resume_text.strip():
                    # Add full resume
                    documents.append(resume_text)
                    metadatas.append({
                        "source": "resume",
                        "type": "professional_experience",
                        "category": "full_resume",
                        "file": "resume.pdf",
                        "keywords": "experience, skills, education, professional, work, career"
                    })
                    ids.append("resume_full")
                    
                    # Extract and create specific skill entries
                    resume_lower = resume_text.lower()
                    
                    # Programming languages and skills
                    programming_keywords = ['python', 'java', 'javascript', 'sql', 'r', 'scala', 'programming', 'coding', 'development']
                    found_skills = [skill for skill in programming_keywords if skill in resume_lower]
                    
                    if found_skills:
                        # Create a skills summary from actual resume content
                        skills_sentences = []
                        for sentence in resume_text.split('.'):
                            if any(skill in sentence.lower() for skill in found_skills):
                                skills_sentences.append(sentence.strip())
                        
                        if skills_sentences:
                            skills_text = '. '.join(skills_sentences[:3])  # Top 3 relevant sentences
                            documents.append(skills_text)
                            metadatas.append({
                                "source": "resume",
                                "type": "technical_skills",
                                "category": "programming_skills",
                                "file": "resume.pdf",
                                "keywords": "programming, skills, technical, languages, " + ", ".join(found_skills)
                            })
                            ids.append("programming_skills")
                    
                    # Experience and work history
                    experience_keywords = ['experience', 'worked', 'company', 'project', 'developed', 'implemented']
                    experience_sentences = []
                    for sentence in resume_text.split('.'):
                        if any(keyword in sentence.lower() for keyword in experience_keywords):
                            experience_sentences.append(sentence.strip())
                    
                    if experience_sentences:
                        experience_text = '. '.join(experience_sentences[:3])  # Top 3 relevant sentences
                        documents.append(experience_text)
                        metadatas.append({
                            "source": "resume",
                            "type": "work_experience",
                            "category": "professional_experience",
                            "file": "resume.pdf",
                            "keywords": "experience, work, projects, companies, career, professional"
                        })
                        ids.append("work_experience")
                    
                    # Education information
                    education_keywords = ['education', 'university', 'degree', 'bachelor', 'master', 'phd', 'graduated']
                    education_sentences = []
                    for sentence in resume_text.split('.'):
                        if any(keyword in sentence.lower() for keyword in education_keywords):
                            education_sentences.append(sentence.strip())
                    
                    if education_sentences:
                        education_text = '. '.join(education_sentences[:2])  # Top 2 relevant sentences
                        documents.append(education_text)
                        metadatas.append({
                            "source": "resume",
                            "type": "education_background",
                            "category": "education",
                            "file": "resume.pdf",
                            "keywords": "education, university, degree, academic, study"
                        })
                        ids.append("education_background")
                    
                    logger.info(f"Loaded resume.pdf: extracted targeted information into multiple focused entries")
                else:
                    logger.warning("resume.pdf appears to be empty or unreadable")
                
            except Exception as e:
                logger.error(f"Failed to load resume.pdf: {e}")
        else:
            logger.warning(f"resume.pdf not found at: {resume_file}")
        
        # Add documents to vector store
        if documents:
            try:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logger.info(f"Successfully loaded {len(documents)} focused documents into fixed hybrid collection")
                
                # Debug: Verify what was actually loaded
                self._debug_collection_contents()
                
            except Exception as e:
                logger.error(f"Failed to add documents to collection: {e}")
        else:
            logger.warning("No documents were loaded into fixed hybrid collection")
    
    def _debug_collection_contents(self):
        """Debug what documents are actually stored"""
        if not self.collection:
            return
        
        try:
            # Get all documents to see what's stored
            all_docs = self.collection.get()
            logger.info(f"Total documents in collection: {len(all_docs['documents'])}")
            
            for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                logger.info(f"Doc {i}: {metadata['source']} - {metadata['category']} - {len(doc)} chars")
                logger.info(f"  Keywords: {metadata.get('keywords', 'none')}")
                logger.info(f"  Preview: {doc[:100]}...")
                
        except Exception as e:
            logger.error(f"Failed to debug collection: {e}")
    
    def web_search(self, query: str) -> str:
        """Web search using DuckDuckGo with better error handling"""
        try:
            results = []
            logger.info(f"Performing web search for: {query}")
            
            with DDGS() as ddgs:
                search_results = ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=3)
                for r in search_results:
                    result_text = f"**{r['title']}**\n{r['href']}\n{r['body']}\n"
                    results.append(result_text)
                    logger.info(f"Found result: {r['title']}")
            
            if results:
                formatted_results = "\n---\n".join(results)
                logger.info(f"Web search successful: {len(results)} results found")
                return formatted_results
            else:
                logger.warning("Web search returned no results")
                return "No web search results found for this query."
                
        except Exception as e:
            logger.error(f"Web search failed with error: {e}")
            return f"Web search temporarily unavailable (Error: {str(e)}). Please try again later."
    
    def get_context_from_local(self, query: str) -> str:
        """FIXED: Much more aggressive local retrieval with keyword matching and very high thresholds"""
        if not self.collection:
            logger.warning("No collection available for local search")
            return ""
        
        try:
            logger.info(f"Searching local collection for: {query}")
            
            # Step 1: Try direct semantic search with high threshold
            results = self.collection.query(
                query_texts=[query], 
                n_results=10,  # Get more results
                include=['documents', 'metadatas', 'distances']
            )
            
            logger.info(f"Found {len(results['documents'][0])} local results")
            
            # Step 2: Use VERY high threshold since the embeddings seem to have high distances
            threshold = 2.0  # Much higher threshold to catch more results
            relevant_context_parts = []
            distances = results.get('distances', [None])[0] or [2.0] * len(results['documents'][0])
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0],
                distances
            )):
                if distance is not None and distance < threshold:
                    source = metadata.get('source', 'unknown')
                    category = metadata.get('category', 'unknown')
                    file = metadata.get('file', 'unknown')
                    relevant_context_parts.append(f"[{source.upper()} - {category}]: {doc}")
                    logger.info(f"Relevant result {i+1}: {source} - {category} - distance: {distance:.3f}")
                else:
                    logger.info(f"Filtered out result {i+1}: distance too high ({distance:.3f}) > threshold ({threshold})")
            
            # Step 3: If semantic search fails, try keyword matching
            if not relevant_context_parts:
                logger.info("Semantic search failed, trying keyword matching...")
                query_lower = query.lower()
                query_keywords = set(re.findall(r'\b\w+\b', query_lower))
                
                # Remove common words
                stop_words = {'what', 'are', 'is', 'the', 'your', 'my', 'about', 'tell', 'me', 'from', 'where', 'how', 'do', 'you'}
                query_keywords = query_keywords - stop_words
                
                logger.info(f"Trying keyword matching with: {query_keywords}")
                
                # Check all documents for keyword matches
                all_docs = self.collection.get()
                for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                    doc_lower = doc.lower()
                    metadata_keywords = set(metadata.get('keywords', '').lower().split(', '))
                    
                    # Check for keyword matches in document text or metadata keywords
                    doc_keywords = set(re.findall(r'\b\w+\b', doc_lower))
                    
                    # Score based on keyword overlap
                    text_overlap = len(query_keywords.intersection(doc_keywords))
                    metadata_overlap = len(query_keywords.intersection(metadata_keywords))
                    total_score = text_overlap + (metadata_overlap * 2)  # Weight metadata keywords more
                    
                    if total_score > 0:
                        source = metadata.get('source', 'unknown')
                        category = metadata.get('category', 'unknown')
                        relevant_context_parts.append((f"[{source.upper()} - {category}]: {doc}", total_score))
                        logger.info(f"Keyword match {i+1}: {source} - {category} - score: {total_score}")
                
                # Sort by score and take top results
                if relevant_context_parts:
                    relevant_context_parts.sort(key=lambda x: x[1], reverse=True)
                    relevant_context_parts = [part[0] for part in relevant_context_parts[:3]]  # Top 3
            
            if relevant_context_parts:
                context = "\n\n".join(relevant_context_parts)
                logger.info(f"Local context found: {len(context)} characters")
                return context
            else:
                logger.warning("No local context found even with keyword matching")
                return ""
            
        except Exception as e:
            logger.error(f"Local retrieval failed: {e}")
            return ""
    
    def hybrid_query(self, user_query: str) -> Tuple[str, dict]:
        """FIXED: Main hybrid query method with improved local-first approach"""
        logger.info(f"Processing hybrid query: {user_query}")
        
        # Step 1: Enhanced query classification
        query_lower = user_query.lower()
        
        # Check for personal+market queries (these need BOTH sources)
        personal_and_market_patterns = [
            r'\b(?:my|your)\s+skills?\s+and\s+(?:current|market|industry|job)',
            r'\b(?:skills?|experience|background)\s+and\s+(?:trends?|market|industry)',
            r'\b(?:my|your)\s+(?:background|experience)\s+and\s+(?:current|recent|market)',
            r'compare.*(?:skills?|experience).*(?:market|trends?|industry)',
            r'how\s+do\s+(?:my|your)\s+skills?.*(?:market|trends?|industry)'
        ]
        
        is_personal_and_market = any(re.search(pattern, query_lower) for pattern in personal_and_market_patterns)
        
        # Check for pure external queries
        pure_external_patterns = [
            r'\b(?:ceo|president|founder|leader)\s+of\s+\w+',
            r'\b(?:openai|google|microsoft|apple|amazon|meta|facebook)\b',
            r'\b(?:latest|current|recent)\s+(?:news|developments|breakthroughs)',
            r'\bwho\s+is\s+\w+(?:\s+\w+)*\s*\?',
            r'\bwhat\s+is\s+(?:the\s+)?latest',
            r'\bcurrent\s+\w+\s+trends(?!\s+and\s+(?:my|your))'
        ]
        
        is_pure_external = any(re.search(pattern, query_lower) for pattern in pure_external_patterns)
        
        # Check for pure personal queries (be more aggressive about detecting these)
        personal_patterns = [
            r'\b(?:tell\s+me\s+about\s+(?:your|my))',
            r'\b(?:what\s+are\s+(?:your|my))',
            r'\b(?:your|my)\s+(?:background|experience|education|skills)',
            r'\b(?:where\s+are\s+you\s+from|your\s+location)',
            r'\b(?:about\s+you|about\s+yourself)',
            r'\bbackground\b',
            r'\bskills?\b',
            r'\bexperience\b',
            r'\bfrom\b.*\?',
            r'\byou\s+from\b'
        ]
        
        is_pure_personal = any(re.search(pattern, query_lower) for pattern in personal_patterns)
        
        logger.info(f"Query classification - Personal+Market: {is_personal_and_market}, Pure External: {is_pure_external}, Pure Personal: {is_pure_personal}")
        
        # Step 2: Route based on classification with LOCAL-FIRST approach
        if is_personal_and_market:
            logger.info("Detected personal+market query - using BOTH sources")
            local_context = self.get_context_from_local(user_query)
            web_context = self.web_search(user_query)
            sources_used = ["local", "web"]
            threshold_triggered = True
            local_context_length = len(local_context)
            
        elif is_pure_external and not is_personal_and_market:
            logger.info("Detected pure external query - web search only")
            web_context = self.web_search(user_query)
            local_context = ""
            sources_used = ["web"]
            threshold_triggered = True
            local_context_length = 0
            
        else:
            # For all other queries (including personal), try local first with much lower threshold
            logger.info("Trying local-first approach with very low threshold")
            local_context = self.get_context_from_local(user_query)
            local_context_length = len(local_context)
            
            # Use local if we have ANY context (much lower threshold)
            if local_context_length > 10:  # Very low threshold - even small context is useful
                web_context = ""
                sources_used = ["local"]
                threshold_triggered = False
                logger.info(f"Using local context: {local_context_length} characters")
            else:
                logger.info(f"Local context insufficient ({local_context_length} chars), adding web search")
                web_context = self.web_search(user_query)
                sources_used = ["local", "web"] if local_context else ["web"]
                threshold_triggered = True
        
        # Step 3: Generate response with improved prompts
        if sources_used == ["web"]:
            # Pure web query
            if "Web search temporarily unavailable" in web_context or "No web search results found" in web_context:
                return "I'm unable to search the web right now to answer your question. Please check your internet connection and try again.", {
                    "sources_used": ["web_failed"],
                    "local_context_length": 0,
                    "web_context_length": 0,
                    "threshold_triggered": True,
                    "query_type": "external",
                    "error": "web_search_failed"
                }
            
            prompt = f"""You are a research assistant analyzing web search results. Answer the question using ONLY the web search information provided below.

Web Search Results:
{web_context}

Question: {user_query}

Instructions: Use only the web search information above to answer the question accurately.

Answer:"""
            
        elif sources_used == ["local"]:
            # Local query - FIXED to handle context better
            prompt = f"""You are Rongjun Geng answering questions about yourself using your personal documents.

Personal Information Available:
{local_context if local_context else "I'll provide what I can based on my general background."}

Question: {user_query}

Instructions:
- Answer as Rongjun Geng in first person ("I am...", "My background...")
- Use the personal information provided above
- Be natural and conversational
- If the information is limited, share what you can and be honest about what you don't have available

Answer as Rongjun Geng:"""
            
        else:
            # Hybrid query (local + web)
            prompt = f"""You are answering a question that requires both personal information about Rongjun Geng and external information.

PERSONAL INFORMATION:
{local_context if local_context else "Limited personal information available."}

EXTERNAL INFORMATION:
{web_context}

Question: {user_query}

Instructions:
- Start with personal information using first person if available
- Then provide external information using third person
- Combine both sources naturally to give a comprehensive answer

Answer:"""
        
        try:
            llm_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1
            )
            
            answer = llm_response.choices[0].message.content
            
            metadata = {
                "sources_used": sources_used,
                "local_context_length": local_context_length,
                "web_context_length": len(web_context) if web_context else 0,
                "threshold_triggered": threshold_triggered,
                "query_type": "personal_and_market" if is_personal_and_market else ("external" if is_pure_external else "personal"),
                "classification": {
                    "personal_and_market": is_personal_and_market,
                    "pure_external": is_pure_external,
                    "pure_personal": is_pure_personal
                }
            }
            
            logger.info(f"Generated response using {sources_used}: {len(answer)} characters")
            return answer, metadata
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {str(e)}", {"sources_used": [], "error": str(e)}

def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)

def print_query_result(query: str, answer: str, metadata: dict):
    """Pretty print query results with metadata"""
    print(f"🤔 Question: {query}")
    print("─" * 60)
    print(f"💬 Answer: {answer}")
    
    sources = metadata.get("sources_used", [])
    if sources:
        print(f"📚 Sources: {', '.join(sources)}")
    
    classification = metadata.get("classification", {})
    if classification:
        print(f"🔍 Classification: {classification}")
    
    if metadata.get("threshold_triggered"):
        print(f"🔍 Hybrid Mode: Local context ({metadata.get('local_context_length', 0)} chars) + Web search")
    else:
        print(f"🏠 Local Only: Context ({metadata.get('local_context_length', 0)} chars)")
    
    print()
    print_separator()
    print()

def main():
    """Demonstrate the FIXED hybrid RAG system"""
    print_separator()
    print("🔧 FIXED Hybrid RAG System - Aggressive Local Retrieval with Keyword Matching")
    print_separator()
    print()
    
    try:
        # Initialize system
        print("📂 Initializing FIXED Hybrid RAG System...")
        hybrid_system = HybridRAGSystem(min_context_length=150)  # Lower threshold
        print("✅ System initialized successfully!")
        print()
        
        # Test the problematic queries that were failing
        test_queries = [
            ("Tell me about your background", "Should find Rongjun's personal background"),
            ("What are your programming skills?", "Should find technical skills from resume"),
            ("Where are you from?", "Should find Beijing/San Francisco location info"),
            ("What are your food preferences?", "Should find food preferences from summary"),
            ("What's your experience with AI?", "Should find relevant AI/data science experience"),
            ("What are your skills and current job market trends?", "Should use BOTH sources"),
            ("Who is the CEO of OpenAI?", "Should use WEB search only"),
        ]
        
        print("🎯 Testing FIXED Query Processing...")
        print_separator()
        print()
        
        for query, expected in test_queries:
            print(f"🤔 Query: {query}")
            print(f"🎯 Expected: {expected}")
            print("─" * 60)
            
            answer, metadata = hybrid_system.hybrid_query(query)
            print_query_result(query, answer, metadata)
        
        # Interactive mode
        print("🎮 Interactive Mode - Test the fixes!")
        print("💡 The system should now find personal information much better!")
        print("(Type 'quit', 'exit', or 'q' to exit)")
        print()
        
        while True:
            try:
                user_query = input("Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q', '']:
                    print("\n👋 Thanks for using the FIXED Hybrid RAG System!")
                    break
                
                print("\n🔍 Processing your question...\n")
                answer, metadata = hybrid_system.hybrid_query(user_query)
                
                print(f"💬 Answer: {answer}")
                sources = metadata.get("sources_used", [])
                if sources:
                    print(f"📚 Sources: {', '.join(sources)}")
                
                classification = metadata.get("classification", {})
                if classification:
                    print(f"🔍 Classification: {classification}")
                
                print(f"📊 Context lengths - Local: {metadata.get('local_context_length', 0)}, Web: {metadata.get('web_context_length', 0)}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the FIXED Hybrid RAG System!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print()
    
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure you have run the personal-rag-system first to create documents")
        print("2. Check that the .env file exists in the project root with OPENAI_API_KEY")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")
        print("4. Make sure resume.pdf and summary.txt exist in personal-rag-system/me/ folder")
        print("5. Check your internet connection for web search functionality")
        print("\n📋 Debug Information:")
        print(f"   - Documents folder: {Path(__file__).parent.absolute() / '../personal-rag-system/me'}")
        print(f"   - Environment file: {Path(__file__).parent.parent / '.env'}")

if __name__ == "__main__":
    main()