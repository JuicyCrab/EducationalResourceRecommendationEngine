"""
This module implements a TF-IDF retriever.

It allows adding documents, computing term frequency (TF), inverse document 
frequency (IDF), and scoring documents based on a query using the TF-IDF algorithm.
"""


from collections import Counter
import math

from retriever_utils import RetrieverUtils


class TFIDFRetriever(RetrieverUtils):
    
    def __init__(self):
        self.documents: dict[int, str] = {}
        self.doc_term_freq: dict[int, Counter] = {} 
        self.doc_freqs: Counter = Counter()
        self.vocab: set[str] = set()
    
    def add_document(self, doc_id: int, text: str):
        self.documents[doc_id] = text
        tokens = super().tokenizer(text)
        
        self.doc_term_freq[doc_id] = Counter(tokens)
        
        unique_terms = set(tokens)
        self.vocab.update(unique_terms)
        for term in unique_terms:
            self.doc_freqs[term] += 1
    
    def compute_tf(self, term: str, doc_id: int) -> float:
        """Computes the log normalized term frequency for a document."""
        
        tf = self.doc_term_freq[doc_id].get(term, 0) #gets the count of a term for this specific doc
        return 1 + math.log(tf) if tf > 0 else 0

    def compute_idf(self, term: str) -> float:
        """Compute the inverse document frequency of a term."""
        total_docs = len(self.documents)
        doc_freq = self.doc_freqs.get(term, 0)
        
        if doc_freq == 0:
            return 0
        
        return math.log(total_docs / doc_freq)
    
    def compute_tfidf_vector(self, doc_id: int) -> dict[str, float]:
        """Compute tf idf vector."""
        vector = {}
        for term in self.doc_term_freq[doc_id]:
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
        """
        Rank all documents against a search query using total TF-IDF weight matching.
        """
        scores = []
        
        for doc_id in self.documents:
            score = self.score_document(query, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
