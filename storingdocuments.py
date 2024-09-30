from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle
import faiss
from langchain.vectorstores import Chroma




def load_documents(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"}, encode_kwargs={'device': "cpu", 'batch_size': 32})
    return embeddings




def create_vector_store(text_chunks, embeddings):
    
    persist_directory = "db"
    vector_store = Chroma.from_documents(text_chunks, embeddings, persist_directory = persist_directory)

    return vector_store

documents = load_documents("data")
text_chunks = split_text_into_chunks(documents)
embeddings = create_embeddings()


vector_store = create_vector_store(text_chunks, embeddings)