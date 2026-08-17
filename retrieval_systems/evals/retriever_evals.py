"""
This class represents the evaluation metrics for retrieval systems. The metrics used
for the task of taking a query and returning relevant documents are precision,
recall, and mean reciprocal rank.
"""

__author__ = "Eyasu Smieja"
__version__ = 1.0


class RetrieverEvals:
    @staticmethod
    def precision(retrieved: list[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
        """Computes the precision using the formula precision = (|tp| / k)"""
        if k == 0:
            raise ValueError("Arg k can't be 0 or negative. Enter a positive number.")
        
        doc_ids = {resource[0] for resource in retrieved[:k]}
        docs_intersected = relevant.intersection(doc_ids)
        numerator = len(docs_intersected)
        denominator = k if len(retrieved) > k else len(retrieved)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

    @staticmethod
    def recall(retrieved: list[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
        """Computes recall using the formula recall = (TP) / (TP + FN)."""
        if len(relevant) == 0:
            raise ValueError("The arg Relevant is an empty set. Enter a set with at least one item. ")
        doc_ids = {resource[0] for resource in retrieved[:k]}
        docs_intersected = relevant.intersection(doc_ids)
        tp = len(docs_intersected)
        fn = len(relevant) - tp
        return tp / (tp + fn)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: list[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
        """Computes mean reciprocal ranking(MRR) using the formula MRR = (1 / N) sum(1 / rank_i)"""
        if len(retrieved) == 0:
            return 0.0
        
        k = len(retrieved) if len(retrieved) < k else k
        for rank in range(0, k):
            if retrieved[rank][0] in relevant:
                return 1 / (rank + 1)
    
        return 0
