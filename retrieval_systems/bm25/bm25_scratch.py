"""
    This script is intended for making the best matching algorithm or BM25 from 
    scratch. Includes computing the term frequency, handling inverted term
    frequency, and utilizing document normalization to prevent term inequality
    for different document sizes. 
"""

import re
from collections import Counter, defaultdict
from math import log
import heapq

def tokenize(text) -> list[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())

def term_frequency(tokens) -> dict[str, int]:
    return Counter(tokens)

def inverted_index(documents) -> dict[str, int]: 
    """
        Takes in tokenized documents and maps the term to the multiple documents. 
        Loop through the docs and get term frequency. 
        if that term frequency doesn't exist as a key than add new key. tf = term frequency 
    """
    inverted_index_result = defaultdict(list)
    for idx, doc in enumerate(documents):
        tf = term_frequency(doc)
        for term in tf:
            inverted_index_result[term].append(idx)

    return inverted_index_result


def inverse_document_frequency(total_number_of_docs, inverted_index) -> dict[str, int]:
    """
        Describes the significance of term rarity across documents. 
    """
    idf = {}
    for key, value in inverted_index.items():
        idf[key] = log((total_number_of_docs - len(value) + 0.5) / (len(value) + 0.5) + 1)

    return idf

def bm25_score(query, doc_tokens, doc_len, avg_doc_len, idf_values, k1=1.5, b=0.75) -> float:
    """
    Computes the bm25 score by utilizing the terms in the query and calculating 
    the inverse document frequency. Also, includes experimental constants 
    k1(term frequency saturation) and b(length normalization). k1 controls how much 
    repeated occurrences of a term matter and prevents favoring of documents that 
    greatly mention a term. b is a penalizer for long documents and values resource 
    length integrity. 
    """
    query_tokens = tokenize(query)
    tf = term_frequency(doc_tokens)
    score = 0
    for token in query_tokens:
        if token not in idf_values:
            continue
        score += idf_values[token] * (tf[token] * (k1 + 1)) / (tf[token] + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    return score

def document_ranking(query, documents) -> list[tuple[float, int]]:
    """
    Handles the ranking of documents using the Bm25 scoring computation. Min-Heap
    organizes the ordering which sorts the ranking by score being the highest times 
    by a negative constant. 
    """
    doc_ranking = []
    total_docs = len(documents)
    inverted_idx = inverted_index(documents)
    avg_doc_len = average_doc_len(documents)
    idf = inverse_document_frequency(total_docs,inverted_idx)

    for idx, doc in enumerate(documents):
        doc_tokens = tokenize(doc)
        doc_len = len(doc_tokens)
        doc_score = bm25_score(query, doc_tokens, doc_len, avg_doc_len, idf)
        heapq.heappush(doc_ranking, (-doc_score, idx))

    sorted_ranking_negatives = sorted(doc_ranking, reverse=False)
    sorted_ranking_positives = [(-1 * score, idx) for score, idx in sorted_ranking_negatives]
    return sorted_ranking_positives
    


def average_doc_len(documents) -> float:
    """
    Computes the average length for documents used for the BM score calculation. 
    """
    if len(documents) == 0:
        return 0

    average_sum = 0

    for doc in documents:
        tokens = tokenize(doc)
        average_sum += len(tokens)

    return average_sum / len(documents)


if __name__ == '__main__':
    pass