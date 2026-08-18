"""
This module parses documents and queries.

It provides utility functions to read and parse JSON files containing resources 
and queries, as well as to extract text from resource files.
"""

import json
import os

from dotenv import load_dotenv


class EvalUtils:

    load_dotenv()
    base_path = os.getenv("FILE_PATH", "")

    @staticmethod
    def json_parser(file_path: str) -> dict | list:
        """Read and parse a JSON configuration file from disk."""
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def resource_text_parser(file_path: str) -> str:
        """Read and return the raw text contents of a document file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @classmethod
    def get_resource_texts(cls) -> list[dict]:
        """
        Load and parse the full collection of target document texts.

        Assembles full document entries by loading metadata from resources.json
        and combining it with raw file text contents.
        """
    
        resource_file_path = os.path.join(cls.base_path, "data", "resources.json")
        resources = cls.json_parser(resource_file_path)
        resource_list = []

        for resource in resources:
            text_path = os.path.join(
                cls.base_path, "data", resource["extracted_text_path"]
            )
            extracted_text = cls.resource_text_parser(text_path)
            resource_list.append(
                {
                    "idx": resource["resource_id"],
                    "extracted_text": extracted_text,
                }
            )

        return resource_list

    @classmethod
    def get_resource_queries(
        cls, start_idx: int = 0, end_idx: int = 15
    ) -> list[dict]:
        """Retrieve a specific range of testing queries for system evaluation."""
        queries_file_path = os.path.join(cls.base_path, "data", "queries.json")
        queries = cls.json_parser(queries_file_path)
        
        if start_idx < 0 or end_idx > len(queries):
            raise ValueError(
                f"Invalid start_idx ({start_idx}) or end_idx ({end_idx}) "
                f"for total query length of {len(queries)}."
            )

        resultant_queries = []
        for query in queries[start_idx:end_idx]:
            resultant_queries.append(
                {
                    "query_id": query["query_id"],
                    "query_text": query["query_text"],
                    "query_relevant_resource_ids": query["relevant_resource_ids"],
                }
            )
        return resultant_queries
