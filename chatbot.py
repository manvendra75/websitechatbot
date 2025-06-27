import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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

def load_website(url):
  loader = WebBasedLoader(url)
  data = loader.load()
  return data

url = "https://www.holidayme.com"
website_data = load_website(url)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(website_data)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a Chroma vector store
vectorstore = Chroma.from_documents(splits, embeddings)

#retrieval setup

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

)

def chat_with_website(query):
    result = qa({"question": query})
    return result['answer']

# Example usage
query = "What is the main topic of this website?"
response = chat_with_website(query)
print(f"Human: {query}")
print(f"AI: {response}")
  
