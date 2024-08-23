# Python script to scrape an article given the url of the article and store the extracted text in a file

import os
import requests
import re
from bs4 import BeautifulSoup

# function to get the html source text of the medium article
def get_page():
	global url
	

	url = input("please provide the the URL of a medium article for scrapping: ")
	headers = {
    	'User-Agent': 'Mozilla/5.0'
	}

	res = requests.get(url, headers=headers)

	res.raise_for_status()
	soup = BeautifulSoup(res.text, 'html.parser')
	return soup

# function to remove all the html tags and replace some with specific strings
def clean(text):

	# This list of tags is to be modified according to cases
    rep = {"<br>": "\n", "<br/>": "\n", "<li>":  "\n",
		   "\nSign up\n": "", "\nSign in\n": "", "\nListen\n": "",
		   "\nShare\n": "", "\nFollow\n": "", "\nHelp\n": "",
		   "\nStatus\n": "", "\nAbout\n": "", "\nCareers\n": "",
		   "\nBlog\n": "", "\nPrivacy\n": "", "\nBlog\n": "",
		   "\nTerms\n": "", "\nText to speech\n": "",
		   "\nTeams\n": "", "\nCS Teacher to ...\n": "",
		   "\n--\n": ""}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    text = re.sub('\<(.*?)\>', '', text)
    return text.strip()

# Text collection
def collect_text(soup):
	text = f'url: {url}\n\n'
	para_text = soup.find_all('p')
	for para in para_text:
		text += f"{para.text}\n\n"
	return text

# function to save file in the current directory
def save_file(text):
	name = url.split("/")[-1]
	fname = f'{name}.txt'
	
	with open(fname, "w") as file:
		file.write(clean(text))

	print(f'File saved in directory {fname}')


if __name__ == '__main__':
	text = collect_text(get_page())
	save_file(text)