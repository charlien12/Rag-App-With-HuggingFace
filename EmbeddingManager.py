from sentence_transformers import SentenceTransformer
class EmbeddingManager:
    def __init__(self,model_name="all-miniLM-L6-v2"):
        self.model_name=model_name
        print("loading model...",model_name)
        self.model=SentenceTransformer(self.model_name)
        print("dimension model...",self.model.get_sentence_embedding_dimension())