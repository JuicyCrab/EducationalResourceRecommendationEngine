# This script will extract the text from the resources 

import requests 
from bs4 import BeautifulSoup

def extract_resources_text(link, parser):
    response = requests.get(link)
    raw_text = response.text 
    soup = BeautifulSoup(raw_text, parser)
    print(soup.get_text())
    print(f"title: {soup.title}") 
    print(f"url: {response.url}")
    

extract_resources_text('https://numpy.org/doc/stable/user/absolute_beginners.html', 'html.parser')

# https://numpy.org/doc/stable/user/absolute_beginners.html