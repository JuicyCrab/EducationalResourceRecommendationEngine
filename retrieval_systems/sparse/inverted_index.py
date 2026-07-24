import re
from collections import defaultdict
from retriever_utils import RetrieverUtils

class InvertedIndex(RetrieverUtils):
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.documents: dict[int, str] = {}
        self.doc_len: dict[int, int] = {}
        self.avg_doc_len: float = 0.0
    
    
    def add_document(self, doc_id: int, text: str):
        """Add document to the index."""
        self.documents[doc_id] = text;
        tokens = super().tokenizer(text)
        self.doc_len[doc_id] = len(tokens)
        
        total_length = sum(self.doc_len.values())
        new_avg_doc_len = total_length / len(self.doc_len)
        
        self.avg_doc_len  = new_avg_doc_len
        for token in tokens:
            self.index[token].add(doc_id)
    
    def search(self, query: str) -> set[int]:
        """Find documents with all the query terms."""
        query_tokens = super().tokenizer(query)
        if not query:
            return set()
        
        result = self.index.get(query_tokens[0], set()).copy() #.get() doesn't throw keyError and the second arg is the default value
        
        for token in query_tokens[1:]:
            result &= self.index.get(token, set()) #want to intersected doc ids, so we return docs with all the query terms
            
        return result