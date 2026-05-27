# This script will extract the text from the resources 

import requests 
import fitz #PyMuPDF import 
from bs4 import BeautifulSoup

def extract_resources_text(link, parser):
    response = requests.get(link)
    raw_text = response.text 
    soup = BeautifulSoup(raw_text, parser)
    print(soup.get_text())
    print(f"title: {soup.title}") 
    print(f"url: {response.url}")
    
extract_resources_text("https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent-exercise", "html.parser")

def pdf_text_parser(link):
    response = requests.get(link)
    with fitz.open(response.content, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text() 
    print(text)
