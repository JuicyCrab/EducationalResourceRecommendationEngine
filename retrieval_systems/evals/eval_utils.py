"""
    This module will parse the documents and queries, so they can be used
    to evaluate the retrieval systems.
"""
import os
from dotenv import load_dotenv
import json

class EvalUtils:
    load_dotenv()   
    base_path = os.getenv("FILE_PATH")
    
    @staticmethod
    def json_parser(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
        
    @staticmethod
    def resource_text_parser(file_path):
        with open(file_path, "r") as file:
            text_content = file.read()
            return text_content
    
    @classmethod
    def get_resource_texts(cls):
        resource_file_path = cls.base_path + '/data/resources.json'
        resources = cls.json_parser(resource_file_path)
        resource_list = []
        
        for resource in resources:
            text_path = cls.base_path + '/data/' + resource['extracted_text_path']
            extracted_text = cls.resource_text_parser(text_path)
            resource_list.append({"idx": resource["resource_id"], "extracted_text": extracted_text})

        return resource_list
    
    @classmethod
    def get_resource_queries(cls, start_idx=0, end_idx=15):
        queries_file_path = cls.base_path + '/data/queries.json'
        queries = cls.json_parser(queries_file_path)
        if (start_idx < 0 or end_idx > len(queries)):
            print("Invalid start or end index for queries.")
        else:
            resultant_queries = []
            for query in queries[start_idx: end_idx]:
                resultant_queries.append({
                    "query_id": query["query_id"],
                    "query_text": query["query_text"],
                    "query_relevant_resource_ids": query["relevant_resource_ids"]
                })
            return resultant_queries