"""
This module implements the Reciprocal Rank Fusion (RRF) algorithm.

It combines the results from two different retrieval systems (BM25 and dense retriever).
"""


from collections import defaultdict


class ReciprocalRankFusion:
    
    @staticmethod
    def search(bm25_results, dense_results, k: int=60) -> list[tuple[str, float]]:
        """
        Combine BM25 sparse results and dense retriever ranks using RRF.

        The reciprocal rank fusion formula follows as: 
        ∑ (1 / (k + rank_retriever(i))), where k is a positive constant.
        """
        docs = defaultdict(int)
        n_bm25 = len(bm25_results)
        n_dense = len(dense_results)
        
        for idx in range(0, n_bm25):
            doc_id_bm25 = bm25_results[idx][0]
            docs[doc_id_bm25] = docs[doc_id_bm25] + (1 / (k + idx + 1))
        
        for idx in range(0, n_dense):
            doc_id_dense = dense_results[idx][0]
            docs[doc_id_dense] = docs[doc_id_dense] + (1 / (k + idx + 1))
            
        rrf_results = list(docs.items())
        return sorted(rrf_results, key=lambda x: x[1], reverse=True)
