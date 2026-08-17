from ..dense.dense_retriever import DenseRetriever
from .eval_utils import EvalUtils

def dense_startup():
    dense = DenseRetriever()
    for text in document_texts:
        dense.embed_documents({text[0]: text[1]})
    dense.save_embeddings()

if __name__ =='__main__':
    
    document_texts = EvalUtils.get_resource_texts()
    resource_queries = EvalUtils.get_resource_queries()
    
    dense = DenseRetriever()
    dense.load_embeddings()
    query_1 = "I want to learn machine learning for robotic systems."
    query_2 = "I am a software engineer that wants to learn the math and statistics behind machine learning."
    results = dense.search(query=query_2, top_k=5)
    print(results)