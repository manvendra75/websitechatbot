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

# Page configuration for iframe embedding
st.set_page_config(
    page_title="HolidayMe Assistant", 
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for cleaner iframe
)

# Custom CSS for iframe optimization
st.markdown("""
<style>
    /* Hide Streamlit branding and menu for cleaner iframe */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Optimize for iframe embedding */
    .stApp {
        margin: 0;
        padding: 10px;
    }
    
    /* Compact header */
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Chat container styling */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
        background: #f9fafb;
    }
</style>
""", unsafe_allow_html=True)

# Compact header for iframe
st.markdown("""
<div class="main-header">
    <h2>🏖️ HolidayMe AI Assistant</h2>
    <p>B2B Travel Technology Solutions & Platform Support</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Force reset conversation if it's the old type
if "conversation" in st.session_state and hasattr(st.session_state.conversation, 'memory'):
    st.session_state.conversation = None

def extract_pdf_text(pdf_files):
    """Extract text from uploaded PDF files"""
    text_content = []
    try:
        for pdf_file in pdf_files:
            pdf_reader = PdfReader(pdf_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            
            if pdf_text.strip():  # Only add if there's actual content
                text_content.append({
                    'filename': pdf_file.name,
                    'content': pdf_text
                })
        
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

def create_vectorstore_from_documents(documents):
    """Create vector store from a list of Document objects"""
    if not documents:
        st.warning("No documents available to create vector store")
        return None
    
    try:
        # Add type metadata to website documents if not already set
        for doc in documents:
            if 'type' not in doc.metadata:
                doc.metadata['type'] = 'website'
        
        # Split all documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Log what was processed
        website_docs = sum(1 for doc in documents if doc.metadata.get('type') == 'website')
        pdf_docs = sum(1 for doc in documents if doc.metadata.get('type') == 'pdf')
        total_chunks = len(splits)
        
        st.success(f"✅ Loaded {website_docs} website pages and {pdf_docs} PDF documents ({total_chunks} chunks)")
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_holidayme_response(question):
    """Handle common identity and contact questions with predefined HolidayMe responses"""
    question_lower = question.lower()
    
    # Identity questions
    if any(phrase in question_lower for phrase in ["who are you", "what are you", "are you openai", "are you chatgpt", "are you ai"]):
        return """I'm the HolidayMe AI Assistant, representing HolidayMe - a leading B2B travel technology company. I help travel industry professionals, tourism boards, travel agencies, and other travel businesses understand our technology solutions and platforms. We provide innovative travel technology services to empower travel companies and destinations worldwide. How can I assist you with information about our B2B travel technology solutions?"""
    
    # Contact information questions
    if any(phrase in question_lower for phrase in ["contact", "phone", "number", "call", "email", "reach you"]):
        return """For business inquiries and partnerships, please visit our website at https://www.holidayme.com where you'll find all our contact information, including phone numbers, email addresses, and business contact options. Our business development team is ready to assist you with technology solutions, platform integrations, and partnership opportunities. You can also find our business contact details directly on our website."""
    
    # Office/location questions
    if any(phrase in question_lower for phrase in ["office", "location", "address", "where are you located"]):
        return """HolidayMe has offices and operations across multiple locations to serve our business partners and clients globally. For specific office addresses and locations, please visit https://www.holidayme.com or contact our business development team. We're here to support travel industry professionals worldwide with our technology solutions and platforms."""
    
    return None  # No predefined response, use RAG

def initialize_conversation(vectorstore):
    """Initialize a simple LLM chain to avoid memory issues"""
    if vectorstore is None:
        return None
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Create a prompt template with HolidayMe persona and conversation history
        prompt_template = """You are the HolidayMe AI Assistant, representing HolidayMe - a leading B2B travel technology company. You help travel industry partners, tourism boards, travel agencies, and other travel professionals with information about our technology solutions, platforms, and services.

IMPORTANT IDENTITY GUIDELINES:
- You work for HolidayMe and represent the company
- HolidayMe is a B2B travel technology company, NOT a direct travel booking agency
- Never mention that you are OpenAI, ChatGPT, or any other AI company
- Always respond as a HolidayMe technology consultant/representative
- For business inquiries, refer partners to HolidayMe's official channels
- Be helpful, professional, and knowledgeable about travel technology solutions

HolidayMe Company Information:
- Website: https://www.holidayme.com
- B2B travel technology solutions provider
- Serves tourism boards, travel companies, and industry partners
- Technology platforms for travel businesses, not direct consumer bookings

Use the following context to answer the question. Consider the conversation history for context, but focus on the current question.

Context: {context}

Conversation History:
{chat_history}

Current Question: {question}

Answer as the HolidayMe AI Assistant:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create simple LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        return {"llm_chain": llm_chain, "vectorstore": vectorstore}
    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        return None

# Backend configuration - hardcoded data sources
BACKEND_CONFIG = {
    "website_url": "https://www.holidayme.com",
    "pdf_files": [
        "docs/Holidayme_RAG.pdf",  # HolidayMe service document
        # Add more PDF file paths here
    ]
}

@st.cache_data
def load_backend_pdfs():
    """Load PDFs from backend file system"""
    pdf_documents = []
    
    for pdf_path in BACKEND_CONFIG["pdf_files"]:
        try:
            if os.path.exists(pdf_path):
                # Read PDF file from file system
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                    
                    if pdf_text.strip():
                        # Create Document object
                        doc = Document(
                            page_content=pdf_text,
                            metadata={
                                'source': os.path.basename(pdf_path),
                                'type': 'pdf',
                                'backend_file': True
                            }
                        )
                        pdf_documents.append(doc)
            else:
                st.warning(f"Backend PDF file not found: {pdf_path}")
                
        except Exception as e:
            st.error(f"Error loading backend PDF {pdf_path}: {str(e)}")
    
    return pdf_documents

# Main app logic - automatic loading
if st.session_state.vectorstore is None:
    with st.spinner("🔄 Loading HolidayMe content..."):
        # Load website data
        website_data = load_website(BACKEND_CONFIG["website_url"])
        
        # Load backend PDFs (if any)
        backend_pdfs = load_backend_pdfs()
        
        # Create vector store with website and backend PDF data
        if website_data or backend_pdfs:
            # Combine website data with backend PDFs
            all_documents = []
            if website_data:
                all_documents.extend(website_data)
            if backend_pdfs:
                all_documents.extend(backend_pdfs)
            
            st.session_state.vectorstore = create_vectorstore_from_documents(all_documents)
            st.session_state.processComplete = True
        else:
            st.warning("Unable to load HolidayMe content. Please refresh the page.")

# Initialize conversation if needed
if st.session_state.conversation is None and st.session_state.vectorstore is not None:
    st.session_state.conversation = initialize_conversation(st.session_state.vectorstore)

# Display chat interface
if st.session_state.processComplete:
    # Chat container with custom styling
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input with placeholder for HolidayMe
    user_input = st.chat_input("Ask about our B2B travel technology solutions, platforms, or partnership opportunities...")
    
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
                        # Check for predefined HolidayMe responses first
                        predefined_response = get_holidayme_response(user_input)
                        
                        if predefined_response:
                            # Use predefined response for identity/contact questions
                            response = predefined_response
                            st.write(response)
                            
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response
                            })
                        else:
                            # Use RAG for travel-related questions
                            # Get relevant documents first
                            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
                            source_docs = retriever.get_relevant_documents(user_input)
                            
                            # Combine context from retrieved documents
                            context = "\n\n".join([doc.page_content for doc in source_docs])
                            
                            # Format chat history for the prompt
                            chat_history = ""
                            if st.session_state.chat_history:
                                # Get last 6 messages (3 exchanges) for context
                                recent_history = st.session_state.chat_history[-6:]
                                for msg in recent_history:
                                    role = "Human" if msg["role"] == "user" else "Assistant"
                                    chat_history += f"{role}: {msg['content']}\n"
                            else:
                                chat_history = "No previous conversation."
                            
                            # Get response from LLM chain
                            llm_chain = st.session_state.conversation["llm_chain"]
                            result = llm_chain.run(
                                context=context, 
                                chat_history=chat_history,
                                question=user_input
                            )
                            response = result
                            st.write(response)
                            
                            # Show source information for RAG responses
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

# Optional: Minimal sidebar for admin functions only
# (Remove this entire block for completely clean interface)
if st.secrets.get("SHOW_ADMIN_PANEL", False):  # Only show if admin mode enabled
    with st.sidebar:
        st.header("🔧 Admin Panel")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Reload Content"):
            st.session_state.vectorstore = None
            st.session_state.conversation = None
            st.session_state.processComplete = None
            st.rerun()
        
        st.divider()
        st.caption(f"📊 Data Source: {BACKEND_CONFIG['website_url']}")
        if st.session_state.processComplete:
            st.caption("✅ Content loaded successfully")
        else:
            st.caption("⏳ Loading content...")