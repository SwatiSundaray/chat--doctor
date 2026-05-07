#from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
from langchain.document_loaders import CSVLoader

load_dotenv()

#2. Load the pdf 
from langchain_community.document_loaders import JSONLoader
def build_index():
    jq_schema = ".[] | {instruction: .instruction, input: .input, output: .output}"
    medical_loader = JSONLoader(file_path="D:/OneDrive - Coforge Limited/Documents/llmops/chat_doctor/data/chatdoctor5k.json",jq_schema=jq_schema,text_content=False)
    docs = medical_loader.load()
    medical_csv_loader=CSVLoader(file_path="D:/OneDrive - Coforge Limited/Documents/llmops/chat_doctor/data/format_dataset.csv")
    medical_csv_docs=medical_csv_loader.load()
    final_medical_docs=medical_csv_docs+docs
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=100,separators=["\n\n", "\n", " ", "", ".",",", ";"])
    recursive_tokens = recursive_splitter.split_documents(final_medical_docs)
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_db = FAISS.from_documents(documents = recursive_tokens, embedding = hf_embeddings)
    faiss_db.save_local("vectorstore")
    print("faiss stored")

if __name__ == "__main__":
        build_index()
