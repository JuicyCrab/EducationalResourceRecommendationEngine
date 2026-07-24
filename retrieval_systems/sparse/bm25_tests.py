from retrieval_systems.sparse.bm25_scratch import document_ranking, toString
import os
from dotenv import load_dotenv
import json

load_dotenv()
def json_parser(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def resource_text_parser(file_path):
    with open(file_path, "r") as file:
        text_content = file.read()
        return text_content
    
def document_ranking_tests(query, resource_texts):
    rank_system = document_ranking(query, resource_texts)
    return rank_system

if __name__ == '__main__':
    base = os.getenv("FILE_PATH")
    queries_file_path = base + '/data/queries.json'
    queries = json_parser(queries_file_path)
    query_1 = queries[0]
    query_1_text = query_1['query_text']
    resource_file_path = base + '/data/resources.json'
    resources = json_parser(resource_file_path)
    resource_1_text_path = base + '/data/texts/r001.txt'
    resource_text_1 = resource_text_parser(resource_1_text_path)
   
    resource_text_list = []
    for idx, resource in enumerate(resources):
        text_path = base + '/data/' + resource['extracted_text_path']
        extracted_text = resource_text_parser(text_path)
        resource_text_list.append(extracted_text)
    
    doc_ranking = document_ranking_tests(query_1_text, resource_texts=resource_text_list)
    toString(doc_ranking)