import math
from collections import Counter
from retriever_utils import RetrieverUtils
class TFIDF_Retriever(RetrieverUtils):
    def __init__(self):
        self.documents: dict[int, str] = {}
        self.doc_term_freq: dict[int, Counter] = {} # individual documents tf Counters
        self.doc_freqs: Counter = Counter() # number of docs containing each term
        self.vocab: set[str] = set()
    
    def add_document(self, doc_id: int, text: str):
        """Index a document."""
        self.documents[doc_id] = text
        tokens = super().tokenizer(text)
        
        self.doc_term_freq[doc_id] = Counter(tokens)
        
        unique_terms = set(tokens)
        self.vocab.update(unique_terms)
        for term in unique_terms:
            self.doc_freqs[term] += 1
    
    def compute_tf(self, term: str, doc_id: int) -> float:
        """Computes term frequency with log normalization"""
        tf = self.doc_term_freq[doc_id].get(term, 0) #gets the count of a term for this specific doc
        return 1 + math.log(tf) if tf > 0 else 0

    def compute_idf(self, term: str) -> float:
        """Compute inverse document frequency"""
        total_docs = len(self.documents)
        doc_freq = self.doc_freqs.get(term, 0)
        
        if doc_freq == 0:
            return 0
        
        return math.log(total_docs / doc_freq)
    
    def compute_tfidf_vector(self, doc_id: int) -> dict[str, float]:
        """Compute tf idf vector."""
        vector = {}
        for term in self.documents[doc_id]:
            tf = self.compute_tf(term, doc_id)
            idf = self.compute_idf(term)
            vector[term] = tf * idf
        
        return vector
    
    def score_document(self, query: str, doc_id: int) -> float:
        score = 0.0
        query_tokens = super().tokenizer(query)
        for token in query_tokens:
            tf = self.compute_tf(token, doc_id)
            idf = self.compute_idf(token)
            score += tf * idf
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        scores = []
        
        for doc_id in self.documents:
            score = self.score_document(query, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True) # lambda tells sort to sort based on the score of the tuple
        return scores[:top_k]
