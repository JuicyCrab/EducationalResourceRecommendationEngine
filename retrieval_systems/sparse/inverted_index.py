"""
This module implements a simple inverted index for document retrieval.

It allows adding documents to the index and searching for documents that contain all the terms in a given query.
"""

from collections import defaultdict


from retriever_utils import RetrieverUtils


class InvertedIndex(RetrieverUtils):
    
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.documents: dict[int, str] = {}
        self.doc_len: dict[int, int] = {}
        self.avg_doc_len: float = 0.0
    
    def add_document(self, doc_id: int, text: str):
        self.documents[doc_id] = text
        tokens = super().tokenizer(text)
        self.doc_len[doc_id] = len(tokens)
        
        total_length = sum(self.doc_len.values())
        new_avg_doc_len = total_length / len(self.doc_len)
        
        self.avg_doc_len = new_avg_doc_len
        for token in tokens:
            self.index[token].add(doc_id)
    
    def search(self, query: str) -> set[int]:
        """
        Search for documents containing all terms present in the query.

        Performs a strict intersection search across the inverted index tokens.
        """
        query_tokens = super().tokenizer(query)
        
        if not query_tokens:
            return set()
        
        result = self.index.get(query_tokens[0], set()).copy()
        
        for token in query_tokens[1:]:
            result &= self.index.get(token, set())
            
        return result