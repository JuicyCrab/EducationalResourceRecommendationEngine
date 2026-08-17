from sentence_transformers import SentenceTransformer, util
import numpy as np
from evals.eval_utils import EvalUtils

class DenseRetriever():
    
    def __init__(self, model_name = 'sentence-transformers/multi-qa-mpnet-base-cos-v1', embedding_file_path = 'dense_resource_embeddings.npy'):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings: np.array = np.array([])
        self.embedding_file_path = embedding_file_path
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_id_file_path = 'dense_retriever_embedding_doc_ids.npy'
    
    def embed_documents(self, documents: list[dict[int, str]]):
        """Embed the documents and store the ids with embeddings. Handles when initially adding documents, and
            when there are documents already populating the vector database.
        """
        new_ids = [doc["idx"] for doc in documents]
        new_texts = [doc["extracted_text"] for doc in documents]
        new_embeddings = self.model.encode(new_texts, convert_to_numpy=True)
        
        self.doc_ids.extend(new_ids)
        self.doc_texts.extend(new_texts)
        
        if self.doc_embeddings.size == 0:
            self.doc_embeddings = new_embeddings
        else:
            self.doc_embeddings = np.vstack([self.doc_embeddings, new_embeddings])
    def save_embeddings(self) -> bool:
        """To prevent from recalculating the embeddings"""
        if self.doc_embeddings.size == 0:
            return False
        
        np.save(self.embedding_file_path, self.doc_embeddings)
        np.save(self.doc_id_file_path, self.doc_ids)
        return True
    
    def load_embeddings(self):
        """Load embeddings from disk or similar storage"""
        self.doc_embeddings = np.load(self.embedding_file_path)
        self.doc_ids = np.load(self.doc_id_file_path).tolist()
    
    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Embed query, cosine similarity, and get results from the documents."""
        query_emb = self.model.encode(query)
        if len(self.doc_ids) == 0:
                    return []
        scores = util.cos_sim(query_emb, self.doc_embeddings)[0].cpu().tolist()
        dot_score_pairs = list(zip(self.doc_ids, scores))
        dot_score_pairs = sorted(dot_score_pairs, key=lambda x: x[1], reverse=True)
        return dot_score_pairs[:top_k]
    