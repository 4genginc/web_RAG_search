# AI Personal Assistant Suite

A modular collection of AI-powered tools for personal information management and research. This suite includes both a Personal RAG System for querying personal documents and a Web Search Agent for real-time information retrieval.

## 📁 Project Structure

```
web_and_rag/
├── agent_web_search/
│    ├── agent_web_search.py         # Web search agent implementation
│    └── README.md                   # Agent Web Search documentation
├── personal-rag-system/
│    ├── me/
│    │    ├── summary.txt            # Personal introduction and background
│    │    └── resume.pdf             # Professional resume (CV)
│    ├── rag_multi-docs.py           # Main RAG system implementation
│    └── README.md                   # Personal RAG System documentation
├── hybrid-rag-system/
│    ├── hybrid_rag.py               # Hybrid system combining both approaches
│    └── README.md                   # Hybrid RAG System documentation
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (create this)
└── README.md                       # This file
```

## 🎯 Overview

This modular AI suite provides three complementary approaches to information retrieval and question answering:

### 🏠 Personal RAG System
Query your personal documents using advanced Retrieval-Augmented Generation (RAG) technology. Perfect for answering questions about your background, experience, and personal information stored locally.

**Use Cases:**
- Interview preparation
- Personal information lookup
- Background summaries
- Skills and experience queries

### 🌐 Agent Web Search
Real-time web search agent that finds current information online using DuckDuckGo. Ideal for research and staying up-to-date with latest developments.

**Use Cases:**
- Current news and trends
- Research assistance
- Real-time information lookup
- Market research

### 🔄 Hybrid RAG System
Intelligently combines both local personal documents and web search to provide comprehensive answers. Automatically determines when to use local data, web search, or both.

**Use Cases:**
- Complex queries requiring both personal and public information
- Professional research with personal context
- Comprehensive question answering
- Adaptive information retrieval

## 🧠 How They Work Together

**Independent Operation:** Each system works standalone for specific use cases
**Hybrid Intelligence:** The hybrid system automatically routes queries to the most appropriate source(s)
**Modular Design:** Use any combination based on your needs

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agent_web_rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file in project root
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Set up Personal RAG System**
   ```bash
   cd personal-rag-system
   mkdir -p me
   # Add your summary.txt and resume.pdf to the me/ folder
   python rag_multi-docs.py
   ```

5. **Try Web Search Agent**
   ```bash
   cd ../agent_web_search
   python agent_web_search.py
   ```

## 📦 Dependencies

All required packages are listed in the shared `requirements.txt`:

```txt
# Core AI/ML Dependencies
openai>=1.12.0
chromadb>=0.4.22
python-dotenv>=1.0.0

# RAG System Dependencies
PyPDF2>=3.0.1
pypdf>=4.0.1
pdfplumber>=0.10.0

# Web Search Dependencies
duckduckgo-search>=6.0.0
agents>=0.1.0

# Optional: Development Tools
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
```

## 🛠 Usage

### Personal RAG System

Navigate to the `personal-rag-system/` folder and run:

```bash
cd personal-rag-system
python rag_multi-docs.py
```

**Sample Questions:**
- "Tell me about my background"
- "What are my main technical skills?"
- "What is my education background?"

### Web Search Agent

Navigate to the `agent_web_search/` folder and run:

```bash
cd agent_web_search
python agent_web_search.py
```

**Sample Queries:**
- Latest developments in AI
- Current market trends
- Recent news about specific topics

### Hybrid RAG System

Navigate to the `hybrid-rag-system/` folder and run:

```bash
cd hybrid-rag-system
python hybrid_rag.py
```

**Sample Queries:**
- "Who is the CEO of OpenAI and what is their background?" (triggers web search)
- "Tell me about my background" (uses local documents)
- "What are my skills and current AI trends?" (combines both sources)

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### Customization

Each system can be customized independently:

- **RAG System**: Modify document processing, chunking, or response generation
- **Web Agent**: Adjust search parameters, result count, or search regions

## 💰 Cost Considerations

### RAG System Costs
- **Setup**: ~$0.001 (one-time document embedding)
- **Per Query**: ~$0.005 (embeddings + LLM response)

### Web Search Costs
- **Per Query**: ~$0.01-0.02 (LLM processing of search results)

### Hybrid RAG Costs
- **Local Only Queries**: Same as RAG system (~$0.005)
- **Hybrid Queries**: RAG + Web costs (~$0.015-0.025)
- **Smart Routing**: Minimizes unnecessary web searches

**Monthly Estimate**: $1-15 for regular personal use across all systems

### Cost Optimization Tips
- Use `gpt-4o-mini` instead of `gpt-4o` for 90% cost reduction
- Implement response caching for frequent queries
- Adjust `max_tokens` to control response length

## 🔧 Development

### Project Structure Benefits

✅ **Modular Design**: Each system is independent and can be developed separately  
✅ **Shared Dependencies**: Common packages in single requirements.txt  
✅ **Easy Maintenance**: Update or modify systems without affecting others  
✅ **Scalable**: Easy to add new AI agents or tools  
✅ **Flexible Usage**: Use individual systems or combine via hybrid approach  
✅ **Smart Routing**: Hybrid system automatically chooses optimal sources  

### Adding New Components

1. Create new folder: `new-agent/` in the `agent_web_rag/` directory
2. Add implementation file and README.md
3. Update main README.md with new component info
4. Add any new dependencies to shared requirements.txt

## 🚨 Troubleshooting

### Common Issues

**API Key Not Found**
```bash
# Ensure .env file is in project root
echo "OPENAI_API_KEY=your_key_here" > .env
```

**Module Not Found**
```bash
# Install all dependencies
pip install -r requirements.txt
```

**RAG System: Documents Not Found**
```bash
# Create and populate documents folder
cd personal-rag-system
mkdir -p me
# Add summary.txt and resume.pdf
```

**Web Search: No Results**
```bash
# Check internet connection and try again
# DuckDuckGo search may have rate limits
```

## 🔒 Security & Privacy

- **Personal Documents**: Remain local, never uploaded to external services
- **API Keys**: Stored in local .env file, not committed to version control
- **Web Searches**: Performed through DuckDuckGo (privacy-focused)
- **Data Processing**: All processing happens locally or through OpenAI API

## 📈 Performance Tips

### RAG System
- Use persistent ChromaDB storage for faster startup
- Implement document caching
- Optimize chunk sizes for your content

### Web Search Agent
- Adjust search result count based on needs
- Implement result caching for frequent queries
- Use specific search queries for better results

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes in appropriate module folder
4. Test both systems independently
5. Update relevant documentation
6. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to new functions
- Include error handling
- Update README files for any new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing powerful AI models
- ChromaDB for excellent vector database
- DuckDuckGo for privacy-focused web search
- Python community for robust ecosystem

## 📞 Support

For questions or issues:
1. Check the individual README files in each module
2. Review troubleshooting section above
3. Check OpenAI API documentation
4. Create an issue in the repository

---

*Built with ❤️ using cutting-edge AI technology for personal productivity*# web_rag_search
# web_RAG_search
