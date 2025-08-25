import os
import logging
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load .env variables
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "mxbai-embed-large")
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
COLLECTION = os.getenv("COLLECTION_NAME", "docs")

def load(urls: List[str]) -> None:
    """Load documents from URLs into Chroma vector store."""
    print("📥 Downloading documents from", urls)

    try:
        # Load from the web
        loader = WebBaseLoader(urls)
        documents = loader.load()

        if not documents:
            raise ValueError("No documents were loaded from the provided URLs")

        # Split into chunks
        markdown_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = markdown_splitter.split_documents(documents)

        if not split_docs:
            raise ValueError("No document chunks were created after splitting")

        # Create embeddings
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        # Store in Chroma
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=COLLECTION,
            persist_directory=PERSIST_DIR
        )

        vectorstore.persist()

        print(f"✅ Loaded {len(split_docs)} chunks from {len(documents)} docs into Chroma at '{PERSIST_DIR}'")

    except Exception as e:
        logger.error(f"❌ Error during data loading: {str(e)}")
        raise

if __name__ == "__main__":
    doc_urls = [
        "https://raw.githubusercontent.com/MicrosoftDocs/azure-databases-docs/refs/heads/main/articles/cosmos-db/nosql/vector-search.md",
        "https://raw.githubusercontent.com/MicrosoftDocs/azure-databases-docs/refs/heads/main/articles/cosmos-db/nosql/multi-tenancy-vector-search.md",
    ]
    load(urls=doc_urls)