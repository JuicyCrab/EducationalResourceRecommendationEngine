import re

class RetrieverUtils:
    def __init__(self):
        pass
    
    def tokenizer(self, text: str) -> str:
        text = text.lower()
        return re.findall(r'\b[a-z]+\b', text)