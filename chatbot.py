import os
os.environ["OPENAI_API_KEY"] = "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"
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


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
  
