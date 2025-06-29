import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.schema import Document

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page configuration
st.set_page_config(page_title="Chat with US", page_icon="📚")
st.title("Chat with our website 📚")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = None

# Force reset conversation if it's the old type
if "conversation" in st.session_state and hasattr(st.session_state.conversation, 'memory'):
    st.session_state.conversation = None

def extract_pdf_text(pdf_files):
    """Extract text from uploaded PDF files"""
    text_content = []
    try:
        st.write(f"🔄 Processing {len(pdf_files)} PDF files...")
        for i, pdf_file in enumerate(pdf_files):
            st.write(f"📄 Processing {pdf_file.name}...")
            pdf_reader = PdfReader(pdf_file)
            pdf_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                pdf_text += page_text
                st.write(f"  Page {page_num + 1}: {len(page_text)} characters")
            
            st.write(f"  Total text length: {len(pdf_text)} characters")
            
            if pdf_text.strip():  # Only add if there's actual content
                text_content.append({
                    'filename': pdf_file.name,
                    'content': pdf_text
                })
                st.success(f"✅ Extracted text from {pdf_file.name}")
            else:
                st.warning(f"⚠️ No text found in {pdf_file.name}")
        
        st.write(f"📊 Successfully processed {len(text_content)} PDF files with content")
        return text_content
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return []

def pdf_to_documents(pdf_text_data):
    """Convert PDF text data to LangChain Documents"""
    documents = []
    try:
        for pdf_data in pdf_text_data:
            # Create a Document object with metadata
            doc = Document(
                page_content=pdf_data['content'],
                metadata={
                    'source': pdf_data['filename'],
                    'type': 'pdf'
                }
            )
            documents.append(doc)
        return documents
    except Exception as e:
        st.error(f"Error converting PDF to documents: {str(e)}")
        return []

@st.cache_resource
def load_website(url):
    """Load website content with caching"""
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return data
    except Exception as e:
        st.error(f"Error loading website: {str(e)}")
        return None

def create_vectorstore(_website_data, _pdf_files=None):
    """Create vector store from website data and optional PDF files"""
    st.write(f"🔧 Creating vector store...")
    st.write(f"  Website data: {len(_website_data) if _website_data else 0} documents")
    st.write(f"  PDF files: {len(_pdf_files) if _pdf_files else 0} files")
    
    all_documents = []
    
    # Add website documents if available
    if _website_data:
        for doc in _website_data:
            # Add metadata to indicate source
            doc.metadata['type'] = 'website'
            all_documents.append(doc)
    
    # Add PDF documents if available
    if _pdf_files:
        try:
            # Extract text from PDFs
            pdf_text_data = extract_pdf_text(_pdf_files)
            st.write(f"📄 Extracted text from {len(pdf_text_data)} PDF files")
            
            # Show preview of PDF content for debugging
            for i, pdf_data in enumerate(pdf_text_data[:2]):  # Show first 2 PDFs
                preview = pdf_data['content'][:500] + "..." if len(pdf_data['content']) > 500 else pdf_data['content']
                st.text(f"PDF {i+1} preview: {preview}")
            
            # Convert to Document objects
            pdf_documents = pdf_to_documents(pdf_text_data)
            all_documents.extend(pdf_documents)
        except Exception as e:
            st.error(f"Error processing PDF files: {str(e)}")
    
    if not all_documents:
        st.warning("No documents available to create vector store")
        return None
    
    try:
        # Split all documents (website + PDF)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create vector store with combined content
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Log what was processed with detailed info
        website_count = len(_website_data) if _website_data else 0
        pdf_count = len(_pdf_files) if _pdf_files else 0
        total_chunks = len(splits)
        
        # Count chunks by type
        website_chunks = sum(1 for chunk in splits if chunk.metadata.get('type') == 'website')
        pdf_chunks = sum(1 for chunk in splits if chunk.metadata.get('type') == 'pdf')
        
        st.success(f"✅ Vector store created with {website_count} website docs and {pdf_count} PDF files")
        st.info(f"📊 Total chunks: {total_chunks} (Website: {website_chunks}, PDF: {pdf_chunks})")
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def initialize_conversation(vectorstore):
    """Initialize a simple LLM chain to avoid memory issues"""
    if vectorstore is None:
        return None
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Create a simple prompt template
        prompt_template = """Use the following context to answer the question. If you cannot find the answer in the context, say so.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create simple LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        return {"llm_chain": llm_chain, "vectorstore": vectorstore}
    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        return None

# Main app logic
url = "https://www.holidayme.com"

# Load and process website + PDFs
with st.spinner("Loading content..."):
    if st.session_state.vectorstore is None:
        website_data = load_website(url)
        
        # Create vector store with both website and PDF data
        if website_data or st.session_state.pdf_docs:
            st.session_state.vectorstore = create_vectorstore(
                website_data, 
                st.session_state.pdf_docs
            )
            st.session_state.processComplete = True
        else:
            st.warning("No website content or PDF files available to process")

# Initialize conversation if needed
if st.session_state.conversation is None and st.session_state.vectorstore is not None:
    st.session_state.conversation = initialize_conversation(st.session_state.vectorstore)

# Display chat interface
if st.session_state.processComplete:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me anything about the website...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if st.session_state.conversation and isinstance(st.session_state.conversation, dict):
                        # Get relevant documents first
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
                        source_docs = retriever.get_relevant_documents(user_input)
                        
                        # Debug: Show what was retrieved
                        pdf_docs_found = sum(1 for doc in source_docs if doc.metadata.get('type') == 'pdf')
                        website_docs_found = sum(1 for doc in source_docs if doc.metadata.get('type') == 'website')
                        st.caption(f"🔍 Retrieved: {website_docs_found} website chunks, {pdf_docs_found} PDF chunks")
                        
                        # If no PDF docs found, try searching specifically for PDF content
                        if pdf_docs_found == 0 and st.session_state.vectorstore:
                            # Try a broader search to see if ANY PDF content exists
                            all_docs = st.session_state.vectorstore.similarity_search("", k=20)
                            total_pdf_chunks = sum(1 for doc in all_docs if doc.metadata.get('type') == 'pdf')
                            st.caption(f"📊 Total PDF chunks in vector store: {total_pdf_chunks}")
                        
                        # Combine context from retrieved documents
                        context = "\n\n".join([doc.page_content for doc in source_docs])
                        
                        # Get response from LLM chain
                        llm_chain = st.session_state.conversation["llm_chain"]
                        result = llm_chain.run(context=context, question=user_input)
                        response = result
                        st.write(response)
                        
                        # Show source information
                        if source_docs:
                            with st.expander("📚 Sources"):
                                sources = set()  # Use set to avoid duplicates
                                for doc in source_docs:
                                    source_type = doc.metadata.get('type', 'unknown')
                                    source_name = doc.metadata.get('source', 'Unknown source')
                                    sources.add(f"{source_type.title()}: {source_name}")
                                
                                for source in sorted(sources):
                                    st.text(f"• {source}")
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": response
                        })
                    elif st.session_state.conversation:
                        # Old conversation object detected, reset it
                        st.session_state.conversation = None
                        st.error("Conversation reset. Please refresh the page or ask your question again.")
                        st.rerun()
                    else:
                        st.error("Conversation not initialized. Please refresh the page.")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write(f"Currently chatting with: {url}")
    
    # Option to change URL
    new_url = st.text_input("Enter a different URL:", value=url)
    if st.button("Load New Website"):
        if new_url != url:
            # Reset session state
            st.session_state.vectorstore = None
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.processComplete = None
            st.session_state.pdf_docs = None
            st.rerun()
    
    st.divider()
    
    # PDF Upload Section
    st.header("📄 PDF Documents")
    st.write("Upload PDFs to augment website information")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base"
    )
    
    if uploaded_files and uploaded_files != st.session_state.pdf_docs:
        st.session_state.pdf_docs = uploaded_files
        # Reset vector store when new PDFs are uploaded
        st.session_state.vectorstore = None
        st.session_state.conversation = None
        st.session_state.processComplete = None
        st.rerun()
    
    if st.button("Process PDFs") and st.session_state.pdf_docs:
        with st.spinner("Processing PDF documents..."):
            # Reset vector store to rebuild with PDFs
            st.session_state.vectorstore = None
            st.session_state.conversation = None
            st.session_state.processComplete = None
            st.rerun()
    
    st.divider()
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Display status
    st.header("📊 Status")
    if st.session_state.processComplete:
        st.success("✅ Website loaded and ready!")
    else:
        st.info("⏳ Loading website content...")
    
    # PDF status
    if st.session_state.pdf_docs:
        st.success(f"📄 {len(st.session_state.pdf_docs)} PDF(s) uploaded")
        for pdf in st.session_state.pdf_docs:
            st.text(f"• {pdf.name}")
    else:
        st.info("📄 No PDFs uploaded")