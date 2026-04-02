from data_ingestion import load_all_pdfs,chunks_doc
from EmbeddingManager import EmbeddingManager
# Step 1 Extract the docs
all_pdf_docs=load_all_pdfs()
#print(all_pdf_docs)
# Step 2 Chunks all the docs
chunks=chunks_doc(all_pdf_docs)
print(len(chunks))
# Step 3 Convert it into embedding chunks
embedding_manager=EmbeddingManager()
