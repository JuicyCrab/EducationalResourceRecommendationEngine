import re
from typing import Dict, List, Set
from collections import defaultdict

class invertedIndex:
    def __init__(self):
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self.documents: Dict[int, str] = {}
        self.doc_len: Dict[int, int] = {}
        self.avg_doc_len: float = 0.0
    
    def tokenize(self, text: str) -> str:
        """Tokenize converts strings to lowercase and ensures the words are standalone."""
        text = text.lower()
        return re.findall(r'\b[a-z]+\b', text)
    
    def add_document(self, doc_id: int, text: str):
        """Add document to the index."""
        self.documents[doc_id] = text;
        tokens = self.tokenize(text)
        self.doc_len[doc_id] = len(tokens)
        
        total_length = sum(self.doc_len.values())
        new_avg_doc_len = total_length / len(self.doc_len)
        
        self.avg_doc_len  = new_avg_doc_len
        for token in tokens:
            self.index[token].add(doc_id)
    
    def search(self, query: str) -> Set[int]:
        """Find documents with all the query terms."""
        query_tokens = self.tokenize(query)
        if not query:
            return set()
        
        result = self.index.get(query_tokens[0], set()).copy() #.get() doesn't throw keyError and the second arg is the default value
        
        for token in query_tokens[1:]:
            result &= self.index.get(token, set()) #want to intersected doc ids, so we return docs with all the query terms
            
        return result