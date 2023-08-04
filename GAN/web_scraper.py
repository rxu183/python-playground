import requests
import os
from bs4 import BeautifulSoup


s = requests.Session()
headers = {'Accept': 'text/html'}
#Curate based on Links: 
ACTUAL_TARGET_URLS = ["https://pixabay.com/images/search/castle"]
NUMBER_PAGES = 10

for i in range(NUMBER_PAGES):
    ACTUAL_TARGET_URLS.append("https://pixabay.com/images/search/castle/?pagi=" + str(i))

CLASS_NAME = "link--WHWzm"
IMG_CLASS_NAME = "fileThumb"
def get_soup(url):
    return BeautifulSoup(url, 'html.parser')

def trawl_pixabay(url):
    current = s.get(url, headers=headers)
    current_webpage = get_soup(current.text)
    print(current_webpage.prettify())
    imgs_on_current = current_webpage.find_all('img', srcset=True)
    for img in imgs_on_current:
        os.system('wget -P raw_images/ %s' % img['src'])

def trawl_site(url):
    current = s.get(url, headers=headers)
    current_webpage = get_soup(current.text)
    content_list = []
    post_list = []
    current_webpage = current_webpage.find_all('article', class_='post-card post-card--preview')
    for article in current_webpage:
        content_list.append(article['data-id']) #A list of data-ids
    for id in content_list:
        post_list.append(url + '/post/' + id)
    for post in post_list:
        trawl_post(post)
    
    #print(current_webpage)
    #print(content_list)

def trawl_post(url):
    current = s.get(url, headers=headers)
    current_webpage = get_soup(current.text)
    #print(current_webpage.prettify())
    current_webpage = current_webpage.find_all('a', class_='fileThumb')
    for a_tag in current_webpage:
        os.system('wget -P raw_images/ %s' % a_tag['href'])
    #print(current_webpage)
    
#for url in TARGET_URLS:
#    trawl_site(url)

for url in ACTUAL_TARGET_URLS:
    trawl_pixabay(url)