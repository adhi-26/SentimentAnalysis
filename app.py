import streamlit as st
import pickle as pkl
import pandas as pd
from utils.download_reviews import download_reviews
from utils.preprocessing_text import preprocess_text
import MoviePage
import ReviewsPage

def load_pickles():
    global model
    with open('pickles/SVC3304548969/SVC3304548969_model.sav', 'rb') as f:
        model = pkl.load(f)
    global vectorizer
    with open('pickles/SVC3304548969/SVC3304548969_vectorizer.sav', 'rb') as f:
        vectorizer = pkl.load(f)

@st.cache_data
def get_reviews(id):
    reviews = pd.DataFrame(download_reviews(id))
    processed_reviews = reviews.Review.map(preprocess_text)
    transformed = vectorizer.transform(processed_reviews)
    return reviews, model.predict(transformed)

movies = pd.read_csv('MovieData/metadata.csv', index_col='imdb-id')
load_pickles()

st.title('Movie Reviews')

selected_movieid = st.selectbox(
                        label= 'Select a movie from the dropdown menu',
                        options= movies.index.values,
                        index = 0,
                        format_func=lambda x: f'{movies.loc[x,"title"]} ({int(movies.loc[x,"release-year"])})',
                        placeholder='Select a movie',                    
                        )

if st.button('Show Results'):
    reviews, predictions = get_reviews(selected_movieid)
    MoviePage.RenderPage(movies.loc[selected_movieid], predictions)
    ReviewsPage.RenderPage(reviews)
    


    
    

