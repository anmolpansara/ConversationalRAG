import os
import bs4
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PERSIST_DIRECTORY = "chroma_db"

try:
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    docs=loader.load()

    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("starting document loading and embedding...")

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
    splits=text_splitter.split_documents(docs)
    vector_store=Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    # retriever=vector_store.as_retriever()

    print("document loading and embedding complete.")

except Exception as e:
    print(f"An error occurred: {e}")
    retriever = None


