from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from pdf2image import convert_from_path
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


# file_path = "C:\Users\RakeshRanjanKumar\Rakesh Developments\Python VS Code\ChatsWithPDFs\Heart_Disease.pdf"
loader = PyPDFLoader('.\Heart_Disease.pdf') 
documents = loader.load_and_split()


text_splitter = RecursiveCharacterTextSplitter (chunk_size=1024, chunk_overlap=64) 
texts = text_splitter.split_documents (documents)
print (len (texts))


embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
faiss_index =FAISS.from_documents (texts, embeddings)
faiss_index_name = 'faiss-index-250'
faiss_index.save_local (faiss_index_name)