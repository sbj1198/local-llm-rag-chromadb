import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "mxbai-embed-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3")
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_data")
COLLECTION = os.getenv("COLLECTION_NAME", "docs")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = Ollama(model=CHAT_MODEL)

vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

prompt = ChatPromptTemplate.from_template(
    "Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": lambda x: x["question"]}
    | prompt
    | llm
    | StrOutputParser()
)

print("Enter your questions (type 'exit' to quit).")
while True:
    q = input("[User]: ")
    if q.strip().lower() == "exit":
        break
    answer = chain.invoke({"question": q})
    print(f"[Assistant]: {answer}")