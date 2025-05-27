from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIRECTORY = "chroma_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def create_rag_chain():
    try:
        # Initialize the same embedding model used during indexing
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load the persisted database
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        
        # Verify that vector DB has documents
        collection_count = vectordb._collection.count()
        print(f"Chroma DB contains {collection_count} documents")
        if collection_count == 0:
            print("WARNING: No documents found in Chroma DB. Please run upload_context.py first.")

        # Create a retriever with more documents and search type
        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5},  # Increase to get more context
            search_type="mmr"  # Use Maximum Marginal Relevance for better diversity
        )

        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=GROQ_API_KEY,
            temperature=0.2  # Lower temperature for more factual responses
        )

        # Create contextualizing prompt - keep the same
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Create improved QA prompt that emphasizes using the context
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer the user's question. "
            "The context contains information about AI agents and LLMs from Lilian Weng's blog post. "
            "Base your answer strictly on the provided context. "
            "If the context doesn't contain the information needed to answer the question, "
            "say 'I don't have enough information about that in my context.' "
            "Make your answer concise but informative.\n\n"
            "Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ]
        )

        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # Create question answering chain with explicit instruction to use context
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        # Create the final RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
        
        # Make the chain also return the source documents
        rag_chain = rag_chain | {
            "answer": lambda x: x["answer"],
            "source_documents": lambda x: x.get("context", [])
        }
        
        # Test the retriever to ensure it works
        print("Testing retriever with a sample question...")
        test_docs = retriever.invoke("What are AI agents?")
        print(f"Retrieved {len(test_docs)} documents for test question")
        
        return rag_chain

    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for better debugging
        return None
    

# Create message history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Global variable to store the initialized chain
global_chain = None

def preload_chain():
    """Preload the RAG chain when application starts"""
    global global_chain
    print("Preloading RAG chain...")
    global_chain = create_rag_chain()
    if global_chain:
        print("RAG chain successfully preloaded!")
    else:
        print("Failed to preload RAG chain")
    return global_chain

# Create Gradio chat interface
def respond(message, history, session_id):
    global global_chain
    
    # Use the preloaded chain if available, or initialize if needed
    if not hasattr(respond, "chain"):
        if global_chain:
            print("Using preloaded RAG chain")
            respond.chain = RunnableWithMessageHistory(
                global_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        else:
            print("Preloaded chain not available, creating new chain...")
            rag_chain = create_rag_chain()
            if rag_chain:
                respond.chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
            else:
                return "Error initializing the chatbot. Please check the logs.", history
    
    try:
        print(f"Processing query: {message}")
        # Invoke the chain with session history
        response = respond.chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Get the answer from the response
        answer = response['answer'] if 'answer' in response else "No response generated."
        
        # Log the retrieved docs for debugging
        if 'source_documents' in response:
            print(f"Retrieved {len(response['source_documents'])} documents")
            for i, doc in enumerate(response['source_documents']):
                print(f"Document {i+1} snippet: {doc.page_content[:100]}...")
        
        # Add message pair to history
        history.append((message, answer))
        return "", history
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(f"Error in respond function: {error_message}")
        import traceback
        traceback.print_exc()
        history.append((message, error_message))
        return "", history

# Create Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# RAG Chatbot with Chat History")
    gr.Markdown("Ask questions about the documents in the vector database!")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your question")
    
    # Using a unique ID for session (could be user-specific in a real app)
    session_id = gr.State("default_session")
    
    # Handle responses
    msg.submit(
        respond,
        [msg, chatbot, session_id],
        [msg, chatbot]
    )
    
    # Add a clear button
    clear_btn = gr.Button("Clear chat")
    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    # Preload the chain before launching the interface
    preload_chain()
    demo.launch()