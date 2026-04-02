import os
import chromadb
import uuid
class VectorStoreManager:
    def __init__(self,persist_dir="data/vector_store",collection_name="pdf_documents"):
        self.collection_name=collection_name
        self.persist_dir=persist_dir
        self.collection=None
        self.client=None
        self._initialize_store()
    def _initialize_store(self):
        os.makedirs(self.persist_dir,exist_ok=True)
        #create a client
        self.client=chromadb.PersistentClient(path=self.persist_dir)
        #create the collection
        self.collection=self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description":"vector store collection for pdf embedding in Rag"}
        )
        print("initailized the vector store with collection=",self.collection_name)
        print("docs in collection",self.collection.count())
        
    def add_documents(self,documents,embeddings):
        if len(documents)!=len(embeddings):
            raise ValueError("number of documents is not match number of embeddings")
        ids=[]
        all_metadata=[]
        documents_list=[]
        embeddings_list=[]
        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            doc_id=f"doc_{uuid.uuid4()}"
            ids.append(doc_id)
            metadata=dict(doc.metadata)
            metadata["doc_index"]=i
            metadata["content-length"]=len(doc.page_content)
            all_metadata.append(metadata)
            documents_list.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
            self.collection.add(
                ids=ids,
                metadatas=all_metadata,
                documents=documents_list,
                embeddings=embeddings_list
            )