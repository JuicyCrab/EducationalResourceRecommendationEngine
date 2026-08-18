"""
This module extracts text.

It extracts text from a given link, which can be an HTML page or a PDF document.
The extracted text can then be used for further processing, such as embedding and semantic search.
"""

import io

from bs4 import BeautifulSoup
import fitz
import requests

def extract_resources_text(link, parser="html.parser"):
    response = requests.get(link)
    raw_text = response.text
    soup = BeautifulSoup(raw_text, parser)
    return soup.get_text(), soup.title, response.url


def pdf_text_parser(link):
    response = requests.get(link)
    with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

