"""
    This script is intended for making the best matching algorithm or BM25 from 
    scratch. Includes computing the term frequency, handling inverted term
    frequency, and utilizing document normalization to prevent term inequality
    for different document sizes. 
"""

from collections import Counter, defaultdict
import re 
from math import log 

def tokenize(text) -> list[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())

def term_frequency(tokens) -> dict[str, int]:
    return Counter(tokens)
        
def inverted_index(documents) -> dict[str, int]: 
    """
        Takes in tokenized documents and maps the term to the multiple documents. Loop through the docs and get term frequency. 
        if that term frequency doesn't exist as a key than add new key. tf = term frequency 
    """
    inverted_index_result = defaultdict(list)
    for idx, doc in enumerate(documents):
        tf = term_frequency(doc)
        for term in tf:
            inverted_index_result[term].append(idx)
            
    return inverted_index_result
             

def inverse_document_frequency(total_documents, inverted_index) -> dict[str, int]:
    """
        Describes the significance of term rarity across documents. 
    """
    N = total_documents
    idf = {}
    for key, value in inverted_index.items():
        idf[key] = log((N - len(value) + 0.5) / (len(value) + 0.5) + 1)
    
    return idf 
    

def full_score():
    pass 

def document_ranking():
    pass 

if __name__ == '__main__':
    pass 