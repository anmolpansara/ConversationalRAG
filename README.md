# RAG Chatbot with Chat History

This project implements a Retrieval Augmented Generation (RAG) chatbot that maintains conversation history for more contextual responses. The chatbot uses documents stored in a vector database to answer user queries.

## Features

- Context-aware responses using RAG
- Maintains chat history for follow-up questions
- Preloaded model for fast initial response
- Supports Groq LLM for high-quality responses
- Uses Chroma as vector database for document storage
- Implements history-aware retriever for better context understanding

## Prerequisites

- Python 3.9+
- Groq API key
- Hugging Face access (for embeddings)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your huggingface token
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd RAG_with_chat_histroy
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Main application file with Gradio interface
- `upload_context.py`: Script to upload documents to the vector database
- `chroma_db/`: Directory where the vector database is stored
- `.env`: Environment variables file

## Usage

### 1. Prepare Your Documents

First, you need to upload documents to the vector database using the `upload_context.py` script:

```bash
python upload_context.py
```

This will create a Chroma vector database in the `chroma_db` directory.

### 2. Run the Chatbot

Launch the chatbot with:

```bash
python main.py
```

The application will:
1. Preload the RAG chain
2. Launch a Gradio interface
3. Be ready to answer questions based on the documents in the vector database

### 3. Interacting with the Chatbot

- Type your questions in the text input
- The chatbot will search the vector database for relevant information
- Chat history is maintained for context-aware responses
- Use the "Clear chat" button to reset the conversation

## How it Works

1. **Document Retrieval**: When a question is asked, the system reformulates it as a standalone query considering chat history
2. **Context Fetching**: The query is used to retrieve relevant documents from the vector database
3. **Answer Generation**: The LLM generates an answer based on the retrieved documents and chat history
4. **Response Display**: The answer is displayed to the user in a clean chat format

## Customization

You can modify these parameters in `main.py`:

- `search_kwargs={"k": 5}`: Number of documents to retrieve (increase for more context)
- `search_type="mmr"`: Search type (MMR provides more diverse results)
- `temperature=0.2`: LLM temperature (lower for more factual responses) 
- Model selection: Change `model="meta-llama/llama-4-scout-17b-16e-instruct"` to use a different model

## Troubleshooting

- If you see "WARNING: No documents found in Chroma DB", run the `upload_context.py` script first
- If responses are slow on first startup, the preloading should reduce this after initial load
- Check the console for detailed logs on document retrieval and any errors

