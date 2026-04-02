from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
#load the our documents
def load_all_pdfs():
    folder_pth="data/pdf"
    num_doc=0
    all_docs=[]
    for file_name in os.listdir(folder_pth):
        if file_name.lower().endswith(".pdf"):
            pdf_path=os.path.join(folder_pth,file_name)
            doc=PyPDFLoader(pdf_path)
            doc=doc.load()
            all_docs.extend(doc)
            num_doc+=1
        return all_docs
    
#convert it into chunks
def chunks_doc(documents,chunk_size=1000,chunk_overload=100):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overload,
        
    )
    chunked_docs=text_splitter.split_documents(documents)
    return chunked_docs

