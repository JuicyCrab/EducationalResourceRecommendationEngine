"""
This script implements the Best Matching algorithm (BM25) from scratch.

It handles computing term frequencies, inverted document frequencies, and
utilizes document length normalization to prevent term inequality across
different text lengths.
"""

from collections import Counter, defaultdict
from math import log
import re

def tokenize(text) -> list[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())

def term_frequency(tokens) -> dict[str, int]:
    return Counter(tokens)

def inverted_index(documents) -> dict[str, int]: 
    """
    Map each unique term to the indices of documents containing it.

    Takes tokenized documents and builds a lookup index.
    """
    inverted_index_result = defaultdict(list)
    for idx, doc in enumerate(documents):
        tf = term_frequency(doc)
        for term in tf:
            inverted_index_result[term].append(idx)
    return inverted_index_result


def inverse_document_frequency(total_number_of_docs, inverted_index) -> dict[str, int]:
    """
    Calculate the significance of term rarity across all documents.
    """
    idf = {}
    for key, value in inverted_index.items():
        numerator = total_number_of_docs - len(value) + 0.5
        denominator = len(value) + 0.5
        idf[key] = log((numerator / denominator) + 1)
    return idf

def bm25_score(query_tokens, doc_tokens, doc_len, avg_doc_len, idf_values, k1=1.5, b=0.75) -> float:
    """
    Compute the BM25 relevance score for a single tokenized document.

    Utilizes experimental constants k1 (term frequency saturation) and b 
    (length normalization) to penalize abnormally long texts.
    """
    tf = term_frequency(doc_tokens)
    score = 0.0
    for token in query_tokens:
        if token not in idf_values:
            continue
        length_norm = 1 - b + b * (doc_len / avg_doc_len)
        tf_component = (tf[token] * (k1 + 1)) / (tf[token] + k1 * length_norm)
        score += idf_values[token] * tf_component
    return score

def document_ranking(query: str, documents: list[str]) -> list[tuple[float, int]]:
    """Rank documents by calculating their BM25 score against a search query."""
    tokenized_docs = [tokenize(doc) for doc in documents]
    total_docs = len(documents)
    
    if total_docs == 0:
        return []
    
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / total_docs
    inverted_idx = inverted_index(tokenized_docs)
    idf = inverse_document_frequency(total_docs, inverted_idx)
    query_tokens = tokenize(query)

    doc_ranking = []
    for idx, doc in enumerate(tokenized_docs):
        doc_len = len(doc)
        doc_score = bm25_score(query_tokens, doc, doc_len, avg_doc_len, idf)
        doc_ranking.append((doc_score, idx))

    return sorted(doc_ranking, key=lambda x: x[0], reverse=True)

def print_rankings(doc_ranking: list[tuple[float, int]]):
    """Print the calculated BM25 scores for the generated document ranking."""
    if not doc_ranking:
        print("Invalid Document Ranking. Must enter a non-None document ranking.")
        return 
    
    for score, idx in document_ranking:
        print(f"Resource {idx} BM Score: {score}")