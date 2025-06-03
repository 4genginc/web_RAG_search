# Personal Agent Web Search

An AI-powered web search agent that uses DuckDuckGo to find current information and provides intelligent responses with source citations. Built using an agent-based architecture for real-time research and information retrieval.

## 🎯 Overview

This agent-based system performs real-time web searches to answer questions about current events, trends, and any information that requires up-to-date data. Unlike static knowledge bases, this system provides the latest information available on the web.

## 🚀 Features

- **Real-time Web Search**: Uses DuckDuckGo for current information
- **Agent Architecture**: Intelligent decision-making for search strategies
- **Source Citation**: Automatically provides URLs and references
- **Privacy-Focused**: Uses DuckDuckGo (no tracking)
- **Customizable Results**: Configurable number of search results
- **Error Handling**: Robust fallback mechanisms
- **Cost Efficient**: Optimized for minimal API usage

## 📁 File Structure

```
agent_web_search/
├── agent_web_search.py         # Main web search agent
└── README.md                   # This documentation
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Internet connection for web search

### Dependencies
The agent requires these packages (install from root `requirements.txt`):
```txt
openai>=1.12.0
duckduckgo-search>=6.0.0
python-dotenv>=1.0.0
agents>=1.0.0  # Custom agents library (if available)
```

### Setup
1. **Navigate to the parent directory and install dependencies**
   ```bash
   cd ..  # Go to project root
   pip install -r requirements.txt
   ```

2. **Ensure environment variables are set**
   ```bash
   # Create .env file in project root (if not exists)
   echo "OPENAI_API_KEY=your_openai_api_key_here" > ../.env
   ```

## 🎮 Usage

### Basic Usage
```bash
cd agent_web_search/
python agent_web_search.py
```

### Customizing the Search Query
Edit the `question` variable in `agent_web_search.py`:
```python
def main():
    question = "Your custom search query here"
    # ... rest of the code
```

### Example Queries
- "Latest developments in artificial intelligence 2025"
- "Current stock market trends"
- "Recent news about climate change"
- "Best practices for Python programming 2025"
- "Search for [specific person] professional information"

## 🔧 Configuration

### Search Parameters
You can modify the search behavior in the `web_search` function:

```python
@function_tool
def web_search(query: str) -> str:
    """Searches the web for up-to-date information."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                query, 
                region='wt-wt',           # Worldwide results
                safesearch='Moderate',    # Safe search level
                max_results=3             # Number of results
            ):
                results.append(f"{r['title']}\n{r['href']}\n{r['body']}\n")
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"
```

### Agent Configuration
Modify the agent settings:
```python
research_agent = Agent(
    name="research_assistant",
    instructions="Custom instructions for your agent...",
    tools=[web_search],
    model="gpt-4o",  # or "gpt-4o-mini" for cost efficiency
)
```

## 🌐 Search Regions

DuckDuckGo supports various regions:
- `'wt-wt'`: Worldwide
- `'us-en'`: United States
- `'uk-en'`: United Kingdom
- `'ca-en'`: Canada
- `'au-en'`: Australia

## 💰 Cost Information

### Per Query Costs
- **DuckDuckGo Search**: Free (no API costs)
- **OpenAI LLM**: ~$0.01-0.03 per query (depending on model and response length)

### Model Cost Comparison
| Model | Input Cost | Output Cost | Typical Query Cost |
|-------|------------|-------------|-------------------|
| gpt-4o | $2.50/1M tokens | $10.00/1M tokens | $0.02-0.05 |
| gpt-4o-mini | $0.15/1M tokens | $0.60/1M tokens | $0.001-0.003 |

### Cost Optimization
- Use `gpt-4o-mini` for 94% cost reduction
- Limit `max_results` in search
- Set reasonable token limits

## 📊 Example Output

```
Assistant: Based on my web search about Rongjun Geng's professional information, here's what I found:

**Professional Background:**
Rongjun Geng appears to be a software engineer and data scientist with experience in...

**Career Highlights:**
- Currently working in the San Francisco Bay Area
- Background in software engineering and data science
- Experience with AI and machine learning technologies

**Sources:**
- LinkedIn Profile: https://www.linkedin.com/in/rjgeng
- GitHub Repository: https://github.com/...
- Professional Portfolio: https://...

The search results indicate expertise in [specific areas] and involvement in [specific projects or companies].
```

## 🔧 Customization

### Adding New Search Tools
You can extend the agent with additional search capabilities:

```python
@function_tool
def academic_search(query: str) -> str:
    """Search academic papers and research."""
    # Implementation for academic search
    pass

@function_tool
def news_search(query: str) -> str:
    """Search recent news articles."""
    # Implementation for news-specific search
    pass

# Add to agent
research_agent = Agent(
    name="research_assistant",
    tools=[web_search, academic_search, news_search],
    # ... other settings
)
```

### Custom Instructions
Modify the agent's behavior:
```python
research_agent = Agent(
    name="research_assistant",
    instructions="""
    You are a specialized research assistant focusing on technology and business.
    - Always verify information from multiple sources
    - Prioritize recent information (within last 2 years)
    - Provide detailed analysis of trends
    - Include relevant statistics when available
    """,
    tools=[web_search],
    model="gpt-4o-mini"
)
```

## 🚨 Troubleshooting

### Common Issues

**Search Error: Connection Failed**
```
Search error: [Connection error details]
```
**Solutions:**
- Check internet connection
- Verify DuckDuckGo is accessible
- Try again in a few minutes (rate limiting)

**Agent Framework Not Found**
```
ModuleNotFoundError: No module named 'agents'
```
**Solutions:**
- Ensure you have the agents library installed
- Check if it's a custom library in your environment
- Modify imports if using a different agent framework

**No Results Found**
```
No results found.
```
**Solutions:**
- Try broader search terms
- Check spelling and grammar
- Ensure the topic has web presence

### Performance Issues

**Slow Responses**
- Reduce `max_results` in search
- Use `gpt-4o-mini` for faster processing
- Implement caching for repeated queries

**High Costs**
- Switch to `gpt-4o-mini` model
- Reduce response length limits
- Monitor usage at OpenAI dashboard

## 🔄 Integration with Other Systems

### With Personal RAG System
```python
# Use for current information to supplement personal data
personal_info = rag_system.query("my background")
current_trends = web_agent.search("current trends in my field")
# Compare and analyze
```

### With Data Analysis
```python
# Search for market data
market_data = web_agent.search("current market trends")
# Process with pandas/analysis tools
```

## 📈 Advanced Usage

### Batch Searches
```python
queries = [
    "AI trends 2025",
    "Remote work statistics",
    "Tech industry news"
]

for query in queries:
    result = Runner.run_sync(
        starting_agent=research_agent,
        input=query
    )
    print(f"Query: {query}")
    print(f"Result: {result.final_output}")
    print("-" * 50)
```

### Custom Analysis
```python
def analyze_search_results(query: str) -> dict:
    """Perform search and extract structured data."""
    result = Runner.run_sync(
        starting_agent=research_agent,
        input=f"Search and analyze: {query}"
    )
    
    # Process result for structured data
    return {
        "query": query,
        "summary": result.final_output,
        "timestamp": datetime.now(),
        "sources": extract_urls(result.final_output)
    }
```

## 📚 Additional Resources

- [DuckDuckGo Search API Documentation](https://github.com/deedy5/duckduckgo_search)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Agent Framework Documentation](https://docs.agent-framework.com) (if applicable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add new search capabilities or improve existing ones
4. Test thoroughly with various queries
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the root LICENSE file for details.

---

*Intelligent web search agent for real-time information retrieval* 🌐