from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import pandas as pd
import streamlit as st

def download_reviews(imdbID):
    '''
    input: 
    1. imdb-id of the movie
    2. name of the file where the movie reviews are stored
    3. destination path where the file is to be saved

    output:
    1. movie reviews as a list
    2. destination file path
    '''


    load_more_path = '//*[@id="load-more-trigger"]'
    load_more_identifier_path = '//div/div[@class="ipl-load-more ipl-load-more--loaded-all"]'
    url = f'https://www.imdb.com/title/{imdbID}/reviews?spoiler=hide&sort=reviewVolume&dir=desc&ratingFilter=0'


    options = webdriver.ChromeOptions()
    options.add_argument("--incognito")
    options.add_argument("--disable-extensions")
    options.add_argument("--headless=new")
    browser = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    browser.get(url)
    #browser.maximize_window()
    time.sleep(2)
    
    identifier = browser.find_elements(By.XPATH, load_more_identifier_path)
    load_more_button = browser.find_element(By.XPATH, load_more_path)
    while not identifier:
        try:
            load_more_button.click()
            time.sleep(3)
            identifier = browser.find_elements(By.XPATH, load_more_identifier_path)
        except Exception as e:
            print(e)
            break
    
    titles = browser.find_elements(By.CSS_SELECTOR,'a.title')
    contents = browser.find_elements(By.CSS_SELECTOR,'div.text')
    movie_reviews = []
    num_reviews = len(titles)
    print(num_reviews)

    # total reviews if total reviews <1000
    # 1000 if reviews >1000 and <2000
    # half of total reviews if reviews >2000
    #num_reviews = max(min(1000,len(titles)),int(0.5*len(titles)))

    df = pd.DataFrame()
    #store reviews in a dataFrame
    for i in range(num_reviews):
        movie_reviews.append({'Title': titles[i].text.replace('\n',' '), 'Review': contents[i].text.replace('\n',' ')})
        try:
            df.loc[i, 'Title'] = titles[i].text.replace('\n',' ')
        except Exception as e:
            pass

        try:
            df.loc[i, 'Review'] = contents[i].text.replace('\n',' ')
        except Exception as e:
            pass
        # filename = title.replace(' ','_') + '_Reviews.csv'

     #export to csv file
    # df.to_csv(f'{destination}/{filename}',encoding='utf-8', index=None)
    return movie_reviews

if __name__ == '__main__':
    reviews = download_reviews('tt0004707')