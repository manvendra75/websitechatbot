# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based B2B chatbot that uses OpenAI's GPT-4 and embeddings to create a conversational interface for HolidayMe's travel technology solutions. The application automatically loads content from the HolidayMe website and backend PDF documents, creates vector embeddings of the combined content, and provides a clean chat interface optimized for iframe embedding to assist travel industry professionals with technology solutions and platform information.

## Architecture

The application is built using:
- **Streamlit** - Web interface framework
- **LangChain** - LLM orchestration and custom RAG implementation
- **OpenAI API** - Language model (gpt-4o) and embeddings (text-embedding-3-small)
- **FAISS** - Vector database for storing document embeddings
- **WebBaseLoader** - Website content scraping
- **PyPDF2** - PDF text extraction

The main flow:
1. Load website content using WebBaseLoader from configured URL
2. Load backend PDF files from configured file system paths
3. Combine website and PDF content into unified document collection
4. Split combined content into chunks using RecursiveCharacterTextSplitter
5. Create embeddings and store in FAISS vector database
6. Initialize custom LLM chain with conversation memory
7. Handle user queries through clean chat interface with source attribution

## Running the Application

```bash
streamlit run chatbot.py
```

## Configuration

- OpenAI API key is configured in `.streamlit/secrets.toml` with the key `OPENAI_API_KEY`
- Website URL and PDF files configured in `BACKEND_CONFIG` dictionary in `chatbot.py`
- Admin panel visibility controlled by `SHOW_ADMIN_PANEL` in secrets.toml
- Embedding model: `text-embedding-3-small`
- Chat model: `gpt-4o` with temperature 0.7
- Text splitting: 1000 character chunks with 200 character overlap

### Backend Content Configuration
```python
BACKEND_CONFIG = {
    "website_url": "https://www.holidayme.com",
    "pdf_files": [
        "docs/umrah-services.pdf",
        "docs/dubai-packages.pdf",
        # Add more PDF paths as needed
    ]
}
```

## Key Functions

- `load_website(url)` - Web scraping with caching
- `load_backend_pdfs()` - Load PDF files from configured backend file system paths
- `create_vectorstore_from_documents(documents)` - Vector database creation from document list
- `initialize_conversation(vectorstore)` - Custom LLM chain setup with conversation memory

## Backend Content Management

The application loads content automatically from backend-configured sources:

1. **Website Loading**: Automatically scrapes configured website URL
2. **PDF Loading**: Loads PDF files from configured file system paths
3. **Document Processing**: Converts all content to LangChain Document objects with metadata
4. **Source Tracking**: Documents tagged with `type: 'website'` or `type: 'pdf'` and source information
5. **Combined Vector Store**: All content embedded together in single FAISS database
6. **Source Attribution**: Responses show which sources provided information
7. **Clean Interface**: No upload widgets - purely chat-focused UI

## Session State Management

The app maintains several session state variables:
- `conversation` - Custom LLM chain implementation (dict with llm_chain and vectorstore)
- `chat_history` - List of chat messages with role/content structure
- `processComplete` - Content loading status flag
- `vectorstore` - FAISS vector database instance containing combined content

## Conversation Memory

The application implements custom conversation memory:
- **Context Window**: Last 6 messages (3 Q&A exchanges) included in prompts
- **Manual Management**: No LangChain memory to avoid output key conflicts
- **Source Integration**: Context from both website and PDF documents
- **History Formatting**: Previous exchanges formatted as "Human:" and "Assistant:" entries

## Dependencies

Based on the imports, install these packages:
```bash
pip install streamlit langchain-openai langchain-community langchain faiss-cpu openai tiktoken PyPDF2 beautifulsoup4 requests
```

## Usage Workflow

1. **Automatic Loading**: App automatically loads website and backend PDF content on startup
2. **Ready to Chat**: Clean interface with chat input immediately available
3. **Querying**: Ask questions that can be answered from website or PDF content
4. **Source Review**: Expand "📚 Sources" to see which documents informed the response
5. **Memory**: Follow-up questions will reference previous conversation context
6. **Admin Functions**: Optional admin panel for clearing chat or reloading content

## Technical Notes

- **Backend Management**: All content sources configured in code, not user uploads
- **Clean Interface**: No file upload widgets, optimized for iframe embedding
- **Custom RAG**: Uses manual retrieval + LLM chain instead of ConversationalRetrievalChain
- **Error Handling**: Graceful degradation if backend PDFs cannot be processed
- **Memory Conflicts**: Avoided by using single output key and manual history management
- **Production Ready**: Optimized for embedding as chat widget on websites