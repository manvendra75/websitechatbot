# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based website chatbot that uses Google's Gemini AI and embeddings to create a conversational interface for querying website content. The application scrapes a specified website, creates vector embeddings of the content, and provides a chat interface where users can ask questions about the website's information.

## Architecture

The application is built using:
- **Streamlit** - Web interface framework
- **LangChain** - LLM orchestration and RAG implementation
- **Google Gemini API** - Language model (gemini-1.5-flash) and embeddings (models/embedding-001)
- **Chroma** - Vector database for storing document embeddings
- **WebBaseLoader** - Website content scraping

The main flow:
1. Load website content using WebBaseLoader
2. Split content into chunks using RecursiveCharacterTextSplitter
3. Create embeddings and store in Chroma vector database
4. Initialize ConversationalRetrievalChain with memory
5. Handle user queries through the chat interface

## Running the Application

```bash
streamlit run chatbot.py
```

## Configuration

- Google API key is configured in `.streamlit/secrets.toml` with the key `GOOGLE_API_KEY`
- Default website URL is hardcoded to "https://www.holidayme.com" in `chatbot.py:90`
- Embedding model: `models/embedding-001`
- Chat model: `gemini-1.5-flash` with temperature 0.7
- Text splitting: 1000 character chunks with 200 character overlap

## Key Functions

- `load_website(url)` - Web scraping with caching
- `create_vectorstore(website_data)` - Vector database creation with caching
- `initialize_conversation(vectorstore)` - RAG chain setup with conversation memory

## Session State Management

The app maintains several session state variables:
- `conversation` - The ConversationalRetrievalChain instance
- `chat_history` - List of chat messages
- `processComplete` - Website loading status flag
- `vectorstore` - Chroma vector database instance

## Dependencies

Based on the imports, install these packages:
```bash
pip install streamlit langchain-google-genai langchain-community langchain chromadb
```