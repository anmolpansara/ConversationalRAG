import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIRECTORY = "chroma_db"

def test_chroma_db():
    try:
        print("Testing connection to Chroma DB...")
        
        # Initialize embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load the persisted database
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        
        # Check if the DB has documents
        collection = vectordb._collection
        count = collection.count()
        print(f"Chroma DB contains {count} documents")
        
        if count == 0:
            print("No documents found in the database. Please run upload_context.py first.")
            return
        
        # Test a query
        print("\nTesting a simple query...")
        query = "What are AI agents?"
        results = vectordb.similarity_search(query, k=2)
        
        print(f"Retrieved {len(results)} documents for query: '{query}'")
        
        # Print the results
        for i, doc in enumerate(results):
            print(f"\nDocument {i+1}:")
            print(f"Content (first 200 chars): {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
        
        print("\nChroma DB test completed successfully.")
        
    except Exception as e:
        print(f"Error testing Chroma DB: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chroma_db()
