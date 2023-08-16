import streamlit as st
import pickle as pkl
import pandas as pd
from utils.download_reviews import download_reviews
from utils.preprocessing_text import preprocess_text
import MoviePage
import ReviewsPage
from wordcloud import WordCloud
def load_pickles():
    global model
    with open('pickles/SVC3304548969/SVC3304548969_model.sav', 'rb') as f:
        model = pkl.load(f)
    global vectorizer
    with open('pickles/SVC3304548969/SVC3304548969_vectorizer.sav', 'rb') as f:
        vectorizer = pkl.load(f)

@st.cache_data(show_spinner=False)
def get_reviews(id):
    reviews = pd.DataFrame(download_reviews(id))
    processed_reviews = reviews.Review.map(preprocess_text)
    transformed = vectorizer.transform(processed_reviews)
    reviews_combined = ' '.join(processed_reviews)
    wordcloud = WordCloud(width=1000, height=800, margin=0).generate(reviews_combined)
    return reviews, model.predict(transformed), wordcloud

movies = pd.read_csv('MovieData/metadata.csv', index_col='imdb-id')
load_pickles()
st.set_page_config('Movie Review Analysis', layout='centered', page_icon=":film:")
for _ in range(10):
    st.header('')

st.title('Movie Reviews Analysis')

selected_movieid = st.selectbox(
                        label= 'Select a movie from the dropdown menu',
                        options= movies.index.values,
                        index = 0,
                        format_func=lambda x: f'{movies.loc[x,"title"]} ({int(movies.loc[x,"release-year"])})',
                        placeholder='Select a movie',                    
                        )

if st.button('Show Results'):
    with st.spinner('Downloading Reviews. This might take a while...'):
        reviews, predictions, wordcloud = get_reviews(selected_movieid)
    MoviePage.RenderPage(movies.loc[selected_movieid], predictions, wordcloud)
    ReviewsPage.RenderPage(reviews)
    


    
    

