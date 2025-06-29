# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based website chatbot that uses OpenAI's GPT-4 and embeddings to create a conversational interface for querying website content and uploaded PDF documents. The application scrapes a specified website, processes uploaded PDFs, creates vector embeddings of the combined content, and provides a chat interface where users can ask questions that draw from both website and PDF information.

## Architecture

The application is built using:
- **Streamlit** - Web interface framework
- **LangChain** - LLM orchestration and custom RAG implementation
- **OpenAI API** - Language model (gpt-4o) and embeddings (text-embedding-3-small)
- **FAISS** - Vector database for storing document embeddings
- **WebBaseLoader** - Website content scraping
- **PyPDF2** - PDF text extraction

The main flow:
1. Load website content using WebBaseLoader
2. Extract text from uploaded PDF files using PyPDF2
3. Combine website and PDF content into unified document collection
4. Split combined content into chunks using RecursiveCharacterTextSplitter
5. Create embeddings and store in FAISS vector database
6. Initialize custom LLM chain with conversation memory
7. Handle user queries through the chat interface with source attribution

## Running the Application

```bash
streamlit run chatbot.py
```

## Configuration

- OpenAI API key is configured in `.streamlit/secrets.toml` with the key `OPENAI_API_KEY`
- Default website URL is hardcoded to "https://www.holidayme.com" in `chatbot.py:90`
- Embedding model: `text-embedding-3-small`
- Chat model: `gpt-4o` with temperature 0.7
- Text splitting: 1000 character chunks with 200 character overlap

## Key Functions

- `load_website(url)` - Web scraping with caching
- `extract_pdf_text(pdf_files)` - Extract text from uploaded PDF files using PyPDF2
- `pdf_to_documents(pdf_text_data)` - Convert PDF text to LangChain Document objects
- `create_vectorstore(website_data, pdf_files)` - Vector database creation from combined website and PDF content
- `initialize_conversation(vectorstore)` - Custom LLM chain setup with conversation memory

## PDF Processing

The application supports uploading multiple PDF files to augment website information:

1. **Upload Interface**: Sidebar file uploader accepts multiple PDF files
2. **Text Extraction**: PyPDF2 extracts text from each page of uploaded PDFs
3. **Document Conversion**: PDF text is converted to LangChain Document objects with metadata
4. **Source Tracking**: PDF documents are tagged with `type: 'pdf'` and `source: filename`
5. **Combined Vector Store**: Website and PDF content are embedded together in single FAISS database
6. **Source Attribution**: Responses show which sources (website or specific PDFs) provided information

## Session State Management

The app maintains several session state variables:
- `conversation` - Custom LLM chain implementation (dict with llm_chain and vectorstore)
- `chat_history` - List of chat messages with role/content structure
- `processComplete` - Website and PDF processing status flag
- `vectorstore` - FAISS vector database instance containing combined content
- `pdf_docs` - Currently uploaded PDF files list

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

1. **Initial Setup**: App loads website content automatically from hardcoded URL
2. **PDF Upload**: Use sidebar to upload one or more PDF files 
3. **Processing**: Click "Process PDFs" to rebuild vector store with combined content
4. **Querying**: Ask questions that can be answered from website or PDF content
5. **Source Review**: Expand "📚 Sources" to see which documents informed the response
6. **Memory**: Follow-up questions will reference previous conversation context

## Technical Notes

- **No Caching**: Vector store creation is not cached to allow dynamic PDF updates
- **Custom RAG**: Uses manual retrieval + LLM chain instead of ConversationalRetrievalChain
- **Error Handling**: Graceful degradation if PDFs cannot be processed
- **Memory Conflicts**: Avoided by using single output key and manual history management