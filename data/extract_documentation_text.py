# This script will extract the text from the resources 

import requests 
import fitz #PyMuPDF import 
from bs4 import BeautifulSoup
import json 
import os 
import time 


def extract_resources_text(link, parser="html.parser"):
    response = requests.get(link)
    raw_text = response.text 
    soup = BeautifulSoup(raw_text, parser)
    print(soup.get_text())
    print(f"title: {soup.title}") 
    print(f"url: {response.url}")



extract_resources_text(f'https://scikit-learn.org/stable/modules/outlier_detection.html')

def pdf_text_parser(link):
    response = requests.get(link)
    with fitz.open(response.content, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text() 
    print(text)

# pdf_text_parser("https://cs229.stanford.edu/notes2022fall/main_notes.pdf")


