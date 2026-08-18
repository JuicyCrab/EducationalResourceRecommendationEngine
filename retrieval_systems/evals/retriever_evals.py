"""
This class represents the evaluation metrics for retrieval systems. 

The metrics used for the task of taking a query and returning relevant 
documents are precision, recall, and mean reciprocal rank.
"""


class RetrieverEvals:

    @staticmethod
    def precision(retrieved: list[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
        """Compute the precision at k score for retrieved items."""
        if k <= 0:
            raise ValueError("Arg k must be a positive integer greater than 0.")
        
        if not retrieved:
            return 0.0

        doc_ids = {resource[0] for resource in retrieved[:k]}
        docs_intersected = relevant.intersection(doc_ids)
        
        return len(docs_intersected) / k

    @staticmethod
    def recall(retrieved: list[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
        """Compute the recall at k score for retrieved items."""
        if len(relevant) == 0:
            raise ValueError("The arg Relevant cannot be an empty set.")
            
        doc_ids = {resource[0] for resource in retrieved[:k]}
        docs_intersected = relevant.intersection(doc_ids)
        
        return len(docs_intersected) / len(relevant)

    @staticmethod
    def mean_reciprocal_rank(retrieved: list[tuple[str, float]], relevant: set[str], k: int = 5) -> float:
        """Compute the Reciprocal Rank (RR) score for a single query iteration."""
        if not retrieved:
            return 0.0

        max_rank = min(len(retrieved), k)
        for rank in range(max_rank):
            if retrieved[rank][0] in relevant:
                return 1 / (rank + 1)

        return 0.0
