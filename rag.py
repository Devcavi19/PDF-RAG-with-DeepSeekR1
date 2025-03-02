import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings  # Change to HuggingFace embeddings

# Add OpenRouter API configuration
from dotenv import load_dotenv

# Updated API key handling for OpenRouter
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    st.success("OpenRouter API key loaded from secrets.")
except (FileNotFoundError, KeyError):
    st.error("No OpenRouter API key found in secrets.")
    st.info("Please create a secrets.toml file in the .streamlit directory with your OPENROUTER_API_KEY.")
    OPENROUTER_API_KEY = ""

# Configure for OpenRouter
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = '/workspaces/codespaces-blank/pdfs/'

# Configure embeddings - use HuggingFace embeddings that don't require an API key
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Small model that works well for embeddings
)
vector_store = InMemoryVectorStore(embeddings)

# Configure LLM using OpenRouter
model = ChatOpenAI(
    model="deepseek/deepseek-r1:free",
    temperature=0.1,
    default_headers={
        "HTTP-Referer": "https://github.com/streamlit-app",  # Replace with your actual site
        "X-Title": "PDF-RAG-App"                             # Replace with your actual title
    }
)

def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)  # Ensure directory exists
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)

# Add a check to handle embedding errors more gracefully
def index_docs(documents):
    try:
        vector_store.add_documents(documents)
        st.success("Documents indexed successfully!")
    except Exception as e:
        st.error(f"An error occurred while indexing documents: {e}")
        st.error(f"Error details: {str(e)}")
        st.info("Check your embedding configuration and ensure the model is available.")

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)