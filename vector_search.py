import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "mxbai-embed-large")
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
COLLECTION = os.getenv("COLLECTION_NAME", "docs")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR
)

query = "show me an example of a vector embedding policy"

results = vectorstore.similarity_search_with_score(query, k=5)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(doc.page_content[:200])
    print("---")