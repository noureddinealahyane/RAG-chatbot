import chromadb
import uuid  # For generating unique IDs
from chromadb.config import Settings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="db/")

# Create or load a collection
collection = client.create_collection(name="mycollection3")

# Function to load PDF documents from a directory
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

# Generate unique IDs for each document
def generate_unique_ids(documents):
    return [str(uuid.uuid4()) for _ in documents]

# Load and process documents
documents = load_documents("data")
text_chunks = split_text_into_chunks(documents)
embeddings = create_embeddings()

# Generate IDs
ids = generate_unique_ids(text_chunks)

# Add documents to the collection
collection.add(
    ids=ids,
    embeddings=embeddings,  # Generate embeddings for the text chunks
    documents=text_chunks,  # Text content
)

# Function to load the existing collection
def load_chromadb_collection(collection_name):
    # Load the existing Chroma collection
    vector_store = Chroma(collection_name, embedding_function=embeddings, persist_directory="./db")
    return vector_store

# Example usage to reload the collection
vector_store = load_chromadb_collection('mycollection3')
