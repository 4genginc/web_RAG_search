# Hybrid RAG System

An intelligent system that automatically combines local personal document retrieval with real-time web search to provide comprehensive answers. This hybrid approach leverages the best of both worlds: personal context from local documents and current information from the web.

**Part of the AI Personal Assistant Suite** - See main project README for full setup instructions.

## 🎯 Overview

The Hybrid RAG System intelligently routes queries to the most appropriate information source(s). It starts with local personal documents and automatically adds web search when local context is insufficient, ensuring comprehensive and accurate responses.

## 🧠 How It Works

### Core Logic

1. **Local Retrieval First**: Always starts by searching personal documents
2. **Context Length Check**: If local context < 300 characters, triggers web search
3. **Smart Combination**: Merges local and web information when both are needed
4. **Intelligent Response**: LLM processes combined context for comprehensive answers

### Decision Flow

```
User Query
    ↓
Local Document Search
    ↓
Context Length Check
    ↓
├─ Sufficient (≥300 chars) → Local Only Response
└─ Insufficient (<300 chars) → Local + Web Search → Combined Response
```

## 🚀 Features

- **Automatic Source Selection**: Intelligently chooses local, web, or both sources
- **Length-Based Triggering**: Uses configurable threshold for web search activation
- **Source Attribution**: Clear labeling of information sources used
- **Cost Optimization**: Minimizes API calls through smart routing logic
- **Fallback Protection**: If one source fails, continues with available sources
- **Rich Metadata**: Provides detailed information about decision-making process
- **Flexible Configuration**: Adjustable thresholds and search parameters

## 📁 Module Structure

```
hybrid-rag-system/
├── hybrid_rag.py               # Main hybrid implementation
└── README.md                   # This file
```

## 🛠 Setup & Usage

### Prerequisites

Ensure you have completed the main project setup:
1. Installed dependencies from main `requirements.txt`
2. Created `.env` file with OpenAI API key in project root
3. **Set up Personal RAG System first** to create personal documents

### Running the Hybrid System

```bash
# Navigate to the hybrid system folder
cd hybrid-rag-system

# Run the hybrid RAG system
python hybrid_rag.py
```

## 🎯 Example Queries & Behavior

### Local-Only Queries (Sufficient Context)

**Query**: "Tell me about my background and where I'm from"
- **Local Context**: 393 characters (from summary.txt)
- **Decision**: Local only ✅
- **Sources**: Personal documents
- **Cost**: ~$0.005

### Hybrid Queries (Insufficient Local Context)

**Query**: "Who is the CEO of OpenAI and what is their background?"
- **Local Context**: 0 characters (no local info about OpenAI CEO)
- **Decision**: Local + Web search 🔍
- **Sources**: Personal docs + DuckDuckGo
- **Cost**: ~$0.015-0.025

### Mixed Information Queries

**Query**: "How do my AI skills compare to current industry trends?"
- **Local Context**: Personal skills from resume
- **Web Context**: Current AI industry trends
- **Decision**: Both sources combined 🔄
- **Result**: Personalized comparison with current market

## ⚙️ Configuration

### Adjustable Parameters

```python
# In hybrid_rag.py
HybridRAGSystem(
    documents_folder="../personal-rag-system/me",  # Path to personal docs
    min_context_length=300,                        # Threshold for web search
)

# Web search parameters
max_results=3,           # Number of web search results
region='wt-wt',          # Search region
safesearch='Moderate'    # Safety level
```

### Model Selection

```python
# Choose model for cost vs quality balance
model="gpt-4o-mini",     # Cost-effective (recommended)
model="gpt-4o",          # Higher quality, higher cost
```

### Custom Thresholds

```python
# Adjust when web search triggers
min_context_length=200,  # More aggressive web search
min_context_length=500,  # More conservative, prefer local
```

## 💰 Cost Analysis

### Cost per Query Type

| Query Type | Local Context | Web Search | Total Cost | Example |
|------------|---------------|------------|------------|---------|
| **Local Only** | ✅ | ❌ | ~$0.005 | "My background" |
| **Web Only** | ❌ | ✅ | ~$0.010 | "Current news" |
| **Hybrid** | ✅ | ✅ | ~$0.015 | "My skills vs trends" |

### Monthly Usage Estimates

**Conservative Usage (50% local, 30% hybrid, 20% web):**
- 100 queries/month: ~$0.80
- 300 queries/month: ~$2.40

**Balanced Usage (30% local, 50% hybrid, 20% web):**
- 100 queries/month: ~$1.20
- 300 queries/month: ~$3.60

### Cost Optimization Benefits

✅ **Smart Routing**: Only uses web search when needed  
✅ **Threshold Control**: Adjustable sensitivity to minimize costs  
✅ **Local Preference**: Prioritizes cheaper local retrieval  
✅ **Efficient Models**: Uses gpt-4o-mini by default  

## 🔍 Advanced Usage

### Query Types & Routing Logic

**Personal Information** → Local Only
- "My education background"
- "My work experience" 
- "My food preferences"

**Current Information** → Web Search Triggered
- "Latest AI developments"
- "Current market trends"
- "Recent news about..."

**Mixed Queries** → Hybrid Approach
- "My skills vs industry requirements"
- "How does my experience compare to current job market?"
- "My background and recent developments in my field"

### Metadata Analysis

```python
answer, metadata = hybrid_system.hybrid_query("Your question")

print(f"Sources used: {metadata['sources_used']}")
print(f"Local context: {metadata['local_context_length']} chars")
print(f"Web triggered: {metadata['threshold_triggered']}")
```

## 🚨 Troubleshooting

### Common Issues

**No Local Context Found**
```
Local context length: 0 characters
```
- Ensure Personal RAG System is set up with documents in `me/` folder
- Check document path configuration
- Verify ChromaDB collection has documents

**Web Search Failures**
```
Web search unavailable
```
- Check internet connection
- Verify DuckDuckGo accessibility
- Try simpler search terms

**Hybrid System Not Loading Personal Documents**
```
Failed to setup local RAG
```
- Run Personal RAG System first: `cd ../personal-rag-system && python rag_multi-docs.py`
- Ensure personal documents exist in `../personal-rag-system/me/`

### Performance Issues

**Slow Responses**
- Reduce `max_results` in web search
- Use `gpt-4o-mini` instead of `gpt-4o`
- Implement local caching for frequent queries

**High Costs**
- Increase `min_context_length` to reduce web searches
- Optimize personal documents for better local coverage
- Monitor usage patterns and adjust thresholds

## 🔒 Privacy & Security

### Data Handling
- **Personal Documents**: Processed locally, never sent to web services
- **Web Searches**: Sent to DuckDuckGo (privacy-focused search engine)
- **Combined Processing**: Done through OpenAI API with standard encryption
- **No Data Storage**: No search history or results cached

### Privacy Features
- Local documents remain on your machine
- Web searches use privacy-focused DuckDuckGo
- No user tracking or data collection
- Secure API key handling

## 📈 Performance Optimization

### Local RAG Optimization
- **Document Quality**: Ensure personal documents are comprehensive and well-organized
- **Chunk Size**: Optimize document chunking for better retrieval
- **Embeddings**: Use consistent embedding models across systems

### Web Search Optimization
- **Query Specificity**: More specific queries yield better web results
- **Result Filtering**: Adjust search parameters for domain-specific results
- **Caching**: Implement result caching for repeated queries

### Hybrid Logic Tuning
```python
# Fine-tune thresholds based on your usage patterns
min_context_length=250,  # For more web augmentation
min_context_length=400,  # For more local preference

# Adjust search parameters
max_results=2,           # Fewer results, faster responses
max_results=5,           # More comprehensive coverage
```

## 🎛 Advanced Configuration

### Custom Routing Logic

```python
def custom_routing_logic(query, local_context):
    """Custom logic for when to trigger web search"""
    
    # Always use web for current events
    if any(word in query.lower() for word in ['latest', 'current', '2025', 'recent']):
        return True
    
    # Use web if local context mentions specific topics but lacks detail
    if 'AI' in local_context and len(local_context) < 500:
        return True
    
    # Default length-based logic
    return len(local_context) < 300
```

### Integration with Other Systems

```python
# Use hybrid system as a service
from hybrid_rag import HybridRAGSystem

hybrid = HybridRAGSystem()

def answer_question(question):
    answer, metadata = hybrid.hybrid_query(question)
    return {
        'answer': answer,
        'sources': metadata['sources_used'],
        'confidence': 'high' if len(metadata['sources_used']) > 1 else 'medium'
    }
```

## 🚀 Future Enhancements

### Potential Improvements
- **Multi-source Web Search**: Add Google, Bing, or specialized APIs
- **Intelligent Caching**: Cache web results and learn from usage patterns
- **Context Ranking**: Score and rank different context sources
- **Real-time Learning**: Adapt thresholds based on user feedback
- **Domain Specialization**: Custom routing for different knowledge domains

### Integration Opportunities
- **API Endpoints**: Expose as REST API for other applications
- **Slack/Discord Bots**: Integrate with communication platforms
- **Note-taking Apps**: Connect with Obsidian, Notion, etc.
- **Voice Interfaces**: Add speech-to-text and text-to-speech

## 📚 Dependencies

Uses shared dependencies from project root:

**Core Requirements:**
```txt
openai>=1.12.0
chromadb>=0.4.22
python-dotenv>=1.0.0
duckduckgo-search>=6.0.0
```

**PDF Processing (inherited from Personal RAG):**
```txt
PyPDF2>=3.0.1
pypdf>=4.0.1
pdfplumber>=0.10.0
```

## 🤝 Contributing

### Development Guidelines
1. **Maintain Modularity**: Keep hybrid system independent
2. **Preserve Routing Logic**: Maintain the core length-based decision making
3. **Add Configurability**: Make new features configurable
4. **Document Decisions**: Explain routing and combination logic
5. **Test Extensively**: Verify behavior across different query types

### Testing Scenarios
- Personal-only queries with sufficient context
- Personal queries with insufficient context
- Pure web queries with no local relevance
- Mixed queries requiring both sources
- Error scenarios (API failures, no network, etc.)

## 📞 Support

For issues specific to the hybrid system:

1. **Check Prerequisites**: Ensure Personal RAG System is set up first
2. **Verify Configuration**: Check document paths and API keys
3. **Test Individual Components**: Try Personal RAG and Web Search independently
4. **Review Logs**: Check console output for routing decisions and errors
5. **Adjust Thresholds**: Fine-tune `min_context_length` for your use case

---

*Intelligently combining local knowledge with global information for comprehensive AI assistance*