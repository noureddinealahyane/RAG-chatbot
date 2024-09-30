import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain_community.llms import Ollama
import tempfile
from langchain.vectorstores import Chroma
import os


DATA_DIR = "loaded files"

# Ensure the "data" directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Function to load all documents from the "data" directory
def load_documents(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"}, encode_kwargs={'device': "cpu", 'batch_size': 32})
    return embeddings

def create_vector_store(text_chunks, embeddings):
    vector_store = Chroma.from_documents(text_chunks, embeddings, persist_directory="./db", collection_name="mycollection8")
    return vector_store

# Function to create vector store
def load_chromadb_collection(collection_name):
    # Load the existing Chroma collection
    vector_store = Chroma(collection_name, embedding_function=embeddings, persist_directory="./db")
    return vector_store


# Function to create LLMS model
def create_llms_model():
    llm = Ollama(
        model="gemma2:2b",
        temperature=0
    )
    return llm

# Initialize Streamlit app
st.title("My RAG ChatBot")
st.title("Personalized Research Assistant")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Ask some scientific questions')
st.markdown('<style>h3{color: gray; text-align: center;}</style>', unsafe_allow_html=True)

# Create embeddings
embeddings = create_embeddings()

# Create vector store
vector_store = load_chromadb_collection('mycollection8')

# Create LLMS model
llm = create_llms_model()

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me about anything"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                                              memory=memory)

# File uploader for PDF documents
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Move the uploaded file to the "data" directory
    new_file_path = os.path.join(DATA_DIR, os.path.basename(temp_file_path))
    os.rename(temp_file_path, new_file_path)

    # Reload documents from the "data" directory
    documents = load_documents(DATA_DIR)
    text_chunks = split_text_into_chunks(documents)
    
    # Rebuild the vector store
    vector_store = create_vector_store(text_chunks, embeddings)

    st.success("File uploaded and added to the knowledge base!")

# Define chat function
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Display chat history
reply_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="Ask me a question", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
            message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
