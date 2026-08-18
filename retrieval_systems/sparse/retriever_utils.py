"""
This module provides the retriever utility functions.

It includes a tokenizer function that processes text by converting it to lowercase 
and extracting alphabetic words.
"""
import re


class RetrieverUtils:
    def __init__(self):
        pass
    
    def tokenizer(self, text: str) -> str:
        text = text.lower()
        return re.findall(r'\b[a-z]+\b', text)