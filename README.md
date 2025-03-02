# PDF RAG Application

A Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their content. The application uses local embedding models with HuggingFace and connects to DeepSeek's language model for generating answers.

## Features

- PDF document upload and processing
- Text chunking and embedding using HuggingFace's `all-MiniLM-L6-v2` model
- Semantic search through document content
- Question answering using DeepSeek's R1:free model via OpenRouter
- User-friendly Streamlit interface

## Technical Stack

- **Frontend/Backend**: Streamlit
- **Document Loading**: PDFPlumber
- **Text Splitting**: LangChain's RecursiveCharacterTextSplitter
- **Embedding Model**: HuggingFace's `all-MiniLM-L6-v2` (runs locally)
- **Vector Store**: InMemoryVectorStore
- **LLM**: DeepSeek R1:free (via OpenRouter API)
- **Local LLM Support**: Ollama (installed separately)

## Prerequisites

- Python 3.8+
- Ollama installed locally
- OpenRouter API key (for accessing DeepSeek model)

## Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install streamlit langchain langchain-community langchain-openai python-dotenv pdfplumber
   pip install sentence-transformers  # For HuggingFace embeddings
   ```

3. Set up your OpenRouter API key:
   - Create a `.streamlit/secrets.toml` file with:
     ```toml
     OPENROUTER_API_KEY = "your_openrouter_api_key_here"
     ```

4. Install Ollama (if not already installed):
   - Follow instructions at [Ollama's official website](https://ollama.ai/)

5. Create a directory for PDF uploads:
   ```bash
   mkdir -p /workspaces/codespaces-blank/pdfs/
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run rag.py
   ```

2. Upload a PDF document using the file uploader.

3. After successful indexing, ask questions in the chat input box.

4. The system will retrieve relevant content from the PDF and generate an answer using DeepSeek's model.

## How It Works

1. **Document Processing**: When you upload a PDF, the application extracts text using PDFPlumber.

2. **Text Chunking**: The text is split into manageable chunks with some overlap.

3. **Embedding Generation**: Each chunk is processed by the HuggingFace `all-MiniLM-L6-v2` model to create vector embeddings.

4. **Question Processing**: When you ask a question, it's also converted to an embedding.

5. **Retrieval**: The application finds document chunks most semantically similar to your question.

6. **Answer Generation**: The relevant chunks along with your question are sent to DeepSeek's R1:free model through OpenRouter API to generate the final answer.

## Notes

- The application uses HuggingFace embeddings locally, which don't require an API key.
- DeepSeek's R1:free model is accessed through OpenRouter API.
- All documents are stored in memory and will be lost when the application is restarted.

## License

--