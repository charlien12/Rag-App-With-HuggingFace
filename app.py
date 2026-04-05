from data_ingestion import load_all_pdfs,chunks_doc
from EmbeddingManager import EmbeddingManager
from VectorStoreManager import VectorStoreManager
from RAGRetriever import RAGRetriever
from GenerateResponse import GenerateResponse
# Step 1 Extract the docs
all_pdf_docs=load_all_pdfs()
#print(all_pdf_docs)
# Step 2 Chunks all the docs
chunks=chunks_doc(all_pdf_docs)
print(len(chunks))
# Step 3 Initializing Embedding
embedding_manager=EmbeddingManager()
#step 4 Initialize Vector
vector_store=VectorStoreManager()

texts=[doc.page_content for doc in chunks]
embedding=embedding_manager.generate_embeddings(texts)
vector_store.add_documents(chunks,embedding)
rag_retrieved=RAGRetriever(embedding_manager,vector_store)
llm=GenerateResponse()
result=GenerateResponse().generate_res("what is RAG",rag_retrieved,llm)
print(result)

