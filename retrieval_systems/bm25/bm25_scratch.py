"""
    This script is intended for making the best matching algorithm or BM25 from 
    scratch. Includes computing the term frequency, handling inverted term
    frequency, and utilizing document normalization to prevent term inequality
    for various document sizes. 
"""

from collections import Counter 
import re 

def tokenize(text) -> list[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())

def term_frequency(tokens) -> dict[str, int]:
    return Counter(tokens)
        
def inverted_index() -> dict[str, int]:
    pass 

def inverse_document_frequency() -> dict[str, int]:
    pass 

def full_score():
    pass 

def document_ranking():
    pass 

if __name__ == '__main__':
    pass 