from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


## load pdf and split into chunks

pdf_path = "data/pdfs"

documents = []

for files in os.listdir(pdf_path):
    if files.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_path, files))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages from PDFs")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

## embedings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db  = FAISS.from_documents(chunks, embeddings)
db.save_local("saved_models/faiss_index")

print("✅ FAISS index built successfully")