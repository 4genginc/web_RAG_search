import os
from dotenv import load_dotenv
from openai import OpenAI
from duckduckgo_search import DDGS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your OpenAI API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class WebSearchAgent:
    def __init__(self, model="gpt-4o-mini"):
        """Initialize the web search agent
        
        Args:
            model: OpenAI model to use (gpt-4o-mini recommended for cost efficiency)
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        logger.info(f"WebSearchAgent initialized with model: {model}")
    
    def web_search(self, query: str, max_results: int = 3) -> str:
        """Search the web using DuckDuckGo and return formatted results
        
        Args:
            query: Search query string
            max_results: Number of search results to return (default: 3)
            
        Returns:
            Formatted search results as a string
        """
        try:
            results = []
            logger.info(f"Searching for: {query}")
            
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query, 
                    region='wt-wt',           # Worldwide results
                    safesearch='Moderate',    # Safe search level
                    max_results=max_results   # Number of results
                ):
                    result_text = f"**{r['title']}**\n{r['href']}\n{r['body']}\n"
                    results.append(result_text)
            
            if results:
                logger.info(f"Found {len(results)} search results")
                return "\n---\n".join(results)
            else:
                logger.warning("No search results found")
                return "No results found."
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search error: {str(e)}"
    
    def search_and_analyze(self, query: str, max_results: int = 3) -> dict:
        """Perform web search and analyze results with AI
        
        Args:
            query: User's question/search query
            max_results: Number of search results to retrieve
            
        Returns:
            Dict with 'answer', 'sources', and 'raw_results'
        """
        try:
            # Step 1: Get web search results
            search_results = self.web_search(query, max_results)
            
            if "Search error" in search_results or "No results found" in search_results:
                return {
                    "answer": "I couldn't find any relevant information for your query. Please try rephrasing your question or checking your internet connection.",
                    "sources": [],
                    "raw_results": search_results
                }
            
            # Step 2: Analyze results with AI
            prompt = f"""You are a research assistant who analyzes web search results to answer user questions. 
Based on the search results below, provide a comprehensive and accurate answer.

IMPORTANT INSTRUCTIONS:
- Use ONLY the information from the search results provided
- Always cite your sources by mentioning the website/URL when possible
- If the search results don't contain enough information, say so honestly
- Be objective and factual
- Organize the information clearly

Web Search Results:
{search_results}

User Question: {query}

Please provide a well-structured answer based on the search results above:"""

            # Get AI response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Extract sources from search results
            sources = []
            for line in search_results.split('\n'):
                if line.startswith('http'):
                    sources.append(line)
            
            logger.info(f"Generated AI analysis for query: {query[:50]}...")
            
            return {
                "answer": answer,
                "sources": sources,
                "raw_results": search_results
            }
            
        except Exception as e:
            logger.error(f"Search and analysis failed: {e}")
            return {
                "answer": f"Error processing your request: {str(e)}",
                "sources": [],
                "raw_results": ""
            }
    
    def chat_with_search(self, user_input: str) -> str:
        """Simple chat interface that decides whether to search or respond directly
        
        Args:
            user_input: User's message/question
            
        Returns:
            Response string
        """
        # Keywords that indicate a search is needed
        search_indicators = [
            'latest', 'current', 'recent', 'news', 'trends', 'what is', 'who is',
            'search for', 'find', 'look up', 'information about', '2024', '2025'
        ]
        
        user_lower = user_input.lower()
        needs_search = any(indicator in user_lower for indicator in search_indicators)
        
        if needs_search:
            logger.info("Search triggered by keywords")
            result = self.search_and_analyze(user_input)
            response = result['answer']
            
            if result['sources']:
                response += f"\n\n**Sources:**\n" + "\n".join(result['sources'][:3])
            
            return response
        else:
            # For non-search queries, respond directly
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user", 
                        "content": f"Please respond to this message: {user_input}"
                    }],
                    max_tokens=300,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"I apologize, but I encountered an error: {str(e)}"

def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)

def print_search_result(query: str, result: dict):
    """Pretty print search results"""
    print(f"🔍 Query: {query}")
    print("─" * 60)
    print(f"💬 Answer: {result['answer']}")
    
    if result.get('sources'):
        print(f"\n📚 Sources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"  {i}. {source}")
    
    print()
    print_separator()
    print()

def main():
    """Main function to demonstrate the Web Search Agent"""
    print_separator()
    print("🌐 Web Search Agent - Real-time Information Retrieval")
    print_separator()
    print()
    
    try:
        # Initialize the agent
        print("🚀 Initializing Web Search Agent...")
        agent = WebSearchAgent(model="gpt-4o-mini")  # Use mini for cost efficiency
        print("✅ Agent initialized successfully!")
        print()
        
        # Example queries to demonstrate the system
        sample_queries = [
            "Latest developments in artificial intelligence 2025",
            "Who is the current CEO of OpenAI?",
            "Recent breakthroughs in quantum computing",
            "Current trends in renewable energy",
            "Search for Rongjun Geng professional information"
        ]
        
        print("🎯 Running Sample Queries...")
        print_separator()
        print()
        
        # Process sample queries
        for query in sample_queries:
            result = agent.search_and_analyze(query)
            print_search_result(query, result)
        
        # Interactive mode
        print("🎮 Interactive Mode - Ask anything!")
        print("💡 Tips: Use keywords like 'latest', 'current', 'search for' to trigger web search")
        print("(Type 'quit', 'exit', or 'q' to exit)")
        print()
        
        while True:
            try:
                user_query = input("Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q', '']:
                    print("\n👋 Thanks for using the Web Search Agent!")
                    break
                
                print("\n🔍 Processing your question...\n")
                
                # Use the chat interface for automatic search detection
                response = agent.chat_with_search(user_query)
                print(f"💬 Response: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using the Web Search Agent!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print()
    
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that the .env file exists in the project root with OPENAI_API_KEY")
        print("2. Verify internet connection for web search")
        print("3. Ensure all dependencies are installed: pip install openai duckduckgo-search python-dotenv")

if __name__ == "__main__":
    main()
