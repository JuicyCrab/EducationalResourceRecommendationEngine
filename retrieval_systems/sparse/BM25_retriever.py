import math
from collections import defaultdict, Counter
from retriever_utils import RetrieverUtils

class BM25Retriever(RetrieverUtils):
    def __init__(self, k1 = 1.5, b = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: dict[int, str] = {}
        self.doc_term_freq: dict[int, Counter] = {}
        self.doc_len: dict[int, int] = {}
        self.doc_freqs = Counter()
        self.avg_doc_len: float = 0.0
        self.inverted_index: dict[str, set[int]] = defaultdict(set)
    
    def add_document(self, doc_id: int, text: str):
        """Add document to the index."""
        self.documents[doc_id] = text
        tokens = super().tokenizer(text)
        
        self.doc_len[doc_id] = len(tokens)
        self.doc_term_freq[doc_id] = Counter(tokens)
        
        total_doc_len = sum(self.doc_len.values()) + len(tokens)
        self.avg_doc_len = total_doc_len / len(self.documents)
        
        unique_terms = set(tokens)
        for term in unique_terms:
            self.doc_freqs[doc_id] += 1
            self.inverted_index[term].add(doc_id)
        
        
    def compute_idf(self, term: str) -> float:
        """
        Compute idf using BM25 formula.
        
        IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
        """
        n_docs = len(self.documents)
        doc_freq = len(self.doc_freqs.get(term, 0))
        
        numerator = n_docs - doc_freq + 0.5
        denominator = doc_freq + 0.5
        
        return math.log((numerator / denominator) + 1)
    
    def score_document(self, query: str, doc_id: int) -> float:
        """
        Computer BM25 score for a document.
        
        Score(D, Q) = sum(IDF(q) * (f(q, D) * (k1 + 1)) /
        (f(q, D) + k1 * (1- b + b * |D| / avgdl()))
        """
        query_tokens = super().tokenizer(query)
        term_freq = self.doc_term_freq[doc_id]
        
        score = 0.0
        
        for token in query_tokens:
            if token not in term_freq:
                continue
            
            tf = term_freq[token]
            idf = self.compute_idf(token)
            
            length_norm = 1 - self.b + self.b * (self.doc_len[doc_id] / self.avg_doc_len)
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
            
            score += idf * tf_component
        
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_tokens = super().tokenizer(query)
        scores = []
        
        candidates = set()
        for token in query_tokens:
            candidates.update(self.inverted_index.get(token, set()))

        for doc_id in candidates:
            score = self.score_document(query, doc_id)
            scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]          
            
            