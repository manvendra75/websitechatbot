import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma  # Fixed import

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

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

@st.cache_resource
def create_vectorstore(_website_data):
    """Create vector store from website data"""
    if not _website_data:
        return None
    
    try:
        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(_website_data)
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create vector store
        vectorstore = Chroma.from_documents(splits, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def initialize_conversation(vectorstore):
    """Initialize the conversation chain"""
    if vectorstore is None:
        return None
    
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        # Create conversation chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        
        return qa
    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        return None

# Main app logic
url = "https://www.holidayme.com"

# Load and process website
with st.spinner("Loading website content..."):
    if st.session_state.vectorstore is None:
        website_data = load_website(url)
        if website_data:
            st.session_state.vectorstore = create_vectorstore(website_data)
            st.session_state.processComplete = True
            st.success("Website loaded successfully!")

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
                    if st.session_state.conversation:
                        result = st.session_state.conversation({"question": user_input})
                        response = result['answer']
                        st.write(response)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": response
                        })
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
            st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.conversation:
            # Reset memory in conversation chain
            st.session_state.conversation.memory.clear()
        st.rerun()
    
    # Display status
    st.divider()
    if st.session_state.processComplete:
        st.success("✅ Website loaded and ready!")
    else:
        st.info("⏳ Loading website content...")