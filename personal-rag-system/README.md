# Personal RAG System

A personalized Retrieval-Augmented Generation (RAG) system that creates an AI-powered Q&A interface using personal documents. This system demonstrates advanced RAG implementation with ChromaDB, OpenAI embeddings, and intelligent document processing.

**Part of the AI Personal Assistant Suite** - See main project README for full setup instructions.

## 🎯 Overview

This RAG system transforms personal documents into an interactive AI assistant that can answer questions about background, experience, skills, and personal details. It showcases the practical application of RAG technology for personal knowledge management and professional representation.

## 🚀 Quick Start

```bash
# Navigate to this module (from project root)
cd personal-rag-system

# Create documents folder and add your files
mkdir -p me
# Add your summary.txt and resume.pdf to the me/ folder

# Run the RAG system
python rag_multi-docs.py
```

## 🚀 Features

- **Intelligent Document Processing**: Automatically processes both text and PDF files
- **PDF Text Extraction**: Supports multiple PDF libraries (pdfplumber, pypdf, PyPDF2)
- **Smart Text Chunking**: Optimized chunking for better retrieval from long documents
- **Semantic Search**: Uses OpenAI embeddings for context-aware retrieval
- **Personalized Responses**: Generates first-person answers maintaining natural conversation flow
- **Source Attribution**: Tracks and cites information sources with detailed metadata
- **Interactive Q&A**: Real-time query processing with comprehensive error handling
- **Professional Focus**: Optimized for technical background, experience, and skills queries
- **Robust Error Handling**: Graceful fallbacks and detailed logging
- **Flexible Setup**: Automatic validation and setup guidance

## 📁 Module Structure

```
personal-rag-system/
├── rag_multi-docs.py           # Main RAG system implementation
├── me/
│    ├── summary.txt            # Personal introduction and background
│    └── resume.pdf             # Professional resume (CV)
└── README.md                   # This file
```

## 🛠 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- **Complete main project setup first** (see root README.md)

### Setup

This module uses the shared dependencies and environment configuration from the project root.

1. **Ensure main project is set up**
   ```bash
   # From project root
   pip install -r requirements.txt
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

2. **Create your personal documents**
   ```bash
   # Navigate to this module
   cd personal-rag-system
   
   # Create documents folder (if not exists)
   mkdir -p me
   ```

3. **Add your personal files**
   
   Create your personal files in the `me/` folder:
   
   **me/summary.txt** - Personal introduction example:
   ```
   My name is [Your Name]. I'm a [your profession]. I'm originally from [location], 
   but I moved to [current location] in [year].
   
   [Add personal details, interests, preferences, etc.]
   ```
   
   **me/resume.pdf** - Your professional resume/CV in PDF format

## 📦 Dependencies

This module uses the shared dependencies from the project root `requirements.txt`:

**Core Dependencies:**
- **openai**: For embeddings and chat completions
- **chromadb**: Vector database for document storage and retrieval
- **python-dotenv**: Environment variable management (from root .env)

**PDF Processing:**
- **PyPDF2/pypdf/pdfplumber**: Multiple PDF processing libraries for robustness

All dependencies are installed from the root level. See main project README for complete dependency list.

## 💰 Cost Information

### OpenAI API Costs

This RAG system uses two types of OpenAI API calls:

**Embeddings (Document Processing):**
- Model: `text-embedding-3-small`
- Cost: **$0.00002 per 1K tokens** (very affordable!)

**Chat Completions (Response Generation):**
- Model: `gpt-4o` 
- Cost: **$2.50 per 1M input tokens, $10.00 per 1M output tokens**

### Typical Usage Costs

**One-Time Setup (Document Loading):**
- Personal summary (400 chars ≈ 100 tokens): ~$0.000002
- Resume PDF (3-5 pages ≈ 2,000 tokens): ~$0.00004
- **Total setup cost: < $0.001** (less than 1/10th of a cent)

**Per Query Costs:**
- Query embedding: ~$0.000002 per question
- Context retrieval + LLM response (avg 500 tokens): ~$0.005
- **Cost per question: ~$0.005** (half a cent per query)

### Real-World Usage Examples

| Usage Pattern | Daily Cost | Monthly Cost |
|---------------|------------|--------------|
| Light use (5-10 questions/day) | $0.025-0.05 | $0.75-1.50 |
| Regular use (20-30 questions/day) | $0.10-0.15 | $3.00-4.50 |
| Heavy use (50+ questions/day) | $0.25+ | $7.50+ |

### Cost Optimization Tips

**1. Use Smaller Model (Recommended):**
```python
# In rag_multi-docs.py, change:
model="gpt-4o-mini",  # Instead of "gpt-4o"
```
- **Savings**: ~90% cost reduction
- **New cost per query**: ~$0.0005 (1/20th of a cent)

**2. Reduce Response Length:**
```python
max_tokens=300,  # Instead of 500
```

**3. Optimize Document Processing:**
- Keep documents concise and relevant
- Remove unnecessary content before processing

### Cost Monitoring

**Track Your Usage:**
- Monitor at: https://platform.openai.com/usage
- Set billing alerts in OpenAI dashboard
- Check logs for API call frequency

**Estimated Monthly Costs:**
- **Minimal personal use**: $0.50-2.00/month
- **Regular professional use**: $2.00-10.00/month
- **Heavy daily use**: $10.00-25.00/month

*Note: Costs may vary based on document size, query complexity, and usage patterns.*

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### Document Requirements

- **summary.txt**: Plain text personal introduction
- **resume.pdf**: Professional resume in PDF format
- Documents should be placed in the `me/` subfolder

## 🎮 Usage

### Basic Usage

```bash
# From the personal-rag-system folder
python rag_multi-docs.py
```

### System Flow

1. **Initialization**: Sets up ChromaDB vector store with OpenAI embeddings
2. **Document Discovery**: Automatically finds and validates documents in `me/` folder
3. **PDF Processing**: Extracts text from PDF files using multiple fallback methods
4. **Text Chunking**: Intelligently splits long documents for optimal retrieval
5. **Vector Storage**: Embeds and stores all document chunks with rich metadata
6. **Query Processing**: Performs semantic search and generates contextual responses
7. **Interactive Mode**: Provides real-time Q&A interface

### Advanced Features

- **Multi-Library PDF Support**: Automatically detects and uses available PDF libraries
- **Intelligent Fallbacks**: If one PDF library fails, tries others automatically
- **Rich Metadata**: Tracks source files, document types, and content categories
- **Error Recovery**: Comprehensive error handling with helpful troubleshooting messages
- **Performance Optimization**: Efficient chunking and embedding strategies

### Example Queries

The system can answer questions like:

- "Tell me about your background and where you're from"
- "What are your main technical skills and expertise?"
- "What is your current job and responsibilities?"
- "What companies have you worked for?"
- "What is your education background?"
- "What leadership experience do you have?"

### Sample Output

```
🚀 Personal RAG System
================================================================================

📂 Initializing Personal RAG System...
📖 Loading personal documents...
✅ Documents loaded successfully!

🎯 Running Sample Queries...
================================================================================

🤔 Question: Tell me about the person's background and where they're from
────────────────────────────────────────────────────────────
💬 Answer: I am originally from [Your City], [Your Country]. In [Year], I moved to 
[Current Location]. My background is in [Your Field] and 
[Your Specialization], and I have experience in both fields.
📚 Sources: resume, summary
📄 Files: summary.txt, resume.pdf
================================================================================

🎮 Interactive Mode - Ask anything!
(Type 'quit', 'exit', or 'q' to exit)

Your question: What are your main skills?
🔍 Processing your question...

💬 Answer: My main technical skills include [Your Skills]...
📚 Sources: resume
```

## 🏗 Architecture

### Core Components

1. **PersonalRAGSystem Class**: Main system orchestrator
2. **Vector Store**: ChromaDB with OpenAI embeddings
3. **Document Processor**: Intelligent text chunking and metadata extraction
4. **Query Engine**: Semantic search and response generation
5. **Prompt Engineering**: Personalized prompts for natural responses

### Technical Stack

- **Vector Database**: ChromaDB for document storage and retrieval
- **Embeddings**: OpenAI text-embedding-3-small for semantic understanding
- **LLM**: OpenAI GPT-4o for response generation
- **Processing**: Custom document chunking and metadata management

### Data Flow

```
Personal Documents → Text Chunking → Vector Embeddings → ChromaDB Storage
                                                              ↓
User Query → Embedding → Semantic Search → Context Retrieval → LLM Generation → Response
```

## 🔧 Customization

### Adding New Document Types

```python
def load_additional_documents(self):
    # Add support for new document formats
    # Example: LinkedIn profile, portfolio content, etc.
    pass
```

### Modifying Response Style

```python
def _create_personal_prompt(self, context: str, query: str) -> str:
    # Customize the prompt template for different response styles
    # Example: More formal, casual, technical focus, etc.
    pass
```

### Extending Metadata

```python
metadatas.append({
    "source": "document_type",
    "type": "content_category", 
    "category": "specific_section",
    "date": "creation_date",        # Add temporal information
    "priority": "importance_level"   # Add relevance scoring
})
```

## 🧪 Testing

### Manual Testing

Run sample queries to verify system functionality:

```python
# Test basic retrieval
result = rag_system.query_personal_info("What is your background?")
print(result["answer"])

# Test technical queries
result = rag_system.query_personal_info("What are your main skills?")
print(result["sources"])
```

### Validation Checklist

- [ ] Documents load successfully
- [ ] Embeddings generate without errors
- [ ] Queries return relevant information
- [ ] Sources are properly attributed
- [ ] Responses use first-person perspective
- [ ] Interactive mode functions correctly

## 🚨 Troubleshooting

### Common Issues

**API Key Error**
```
ValueError: OPENAI_API_KEY not found in environment variables
```
Solution: Ensure the `.env` file exists in the **project root** with valid OpenAI API key

**Document Loading Error**
```
Documents folder 'me' not found!
```
Solution: Create the `me/` folder in this module directory and add your documents:
```bash
mkdir -p me
# Add summary.txt and resume.pdf to the me/ folder
```

**No Documents Found**
```
No personal documents found!
```
Solution: Ensure you have at least one of these files in the `me/` folder:
- `me/summary.txt` - Personal introduction
- `me/resume.pdf` - Professional resume

**Embedding Generation Error**
```
HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 401 Unauthorized"
```
Solution: Check API key validity and billing status

**No Results Found**
```
I don't have information about that topic in my knowledge base.
```
Solution: Ensure documents contain relevant information and try rephrasing query

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

- **API Key Protection**: Never commit `.env` files to version control
- **Data Privacy**: Personal documents remain local and are not shared
- **Network Security**: All API calls use HTTPS encryption
- **Access Control**: Consider implementing authentication for production use

## 🚀 Deployment Options

### Local Development
- Run directly with Python interpreter
- Suitable for personal use and testing

### Web Application
- Integrate with Flask/FastAPI for web interface
- Add authentication and session management
- Deploy to cloud platforms (Heroku, AWS, etc.)

### API Service
- Expose as REST API endpoints
- Implement rate limiting and caching
- Add monitoring and analytics

## 📈 Performance Optimization

### Vector Store Optimization
- Use persistent ChromaDB storage for faster startup
- Implement embedding caching to reduce API calls
- Optimize chunk sizes for better retrieval

### Response Optimization
- Cache frequent queries
- Implement response streaming for real-time feel
- Add response quality scoring

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for new functions
- Include error handling and logging

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Module Information

**Personal RAG System**
- Part of the AI Personal Assistant Suite
- Focuses on personal document retrieval and Q&A
- See main project README for full suite overview and setup

## 🔗 Related Modules

- **Agent Web Search**: Real-time web search capabilities
- **Main Project**: Complete AI assistant suite documentation

## 🙏 Acknowledgments

- OpenAI for providing powerful embedding and language models
- ChromaDB team for excellent vector database implementation
- Python community for robust ecosystem and libraries

## 📚 Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [RAG Best Practices](https://python.langchain.com/docs/modules/data_connection/)
- [Vector Database Comparison](https://www.pinecone.io/learn/vector-database/)

---

*Built with ❤️ using RAG technology for personal knowledge management*