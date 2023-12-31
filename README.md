# SentimentAnalysis
Sentiment Analysis on Movie Reviews using Streamlit to build front-end.  
App - https://moviesentimentanalysis.streamlit.app/

# Dataset
Dataset with 25000k reviews for training and testing each with no more than 30 reviews from a movie
https://ai.stanford.edu/~amaas/data/sentiment/  

# Files
1. Datasets : **processed** and **raw** **training** and **test** data  
2. Images : Screenshots from the deployed webpage  
3. MovieData : Metadata of around **70k** movies scraped from **IMDb**  
4. pickles : pickles and **Performance** of ML model and Pointwise Mutual Information(PMI) scores and its details  
5. .gitignore  
6. MoviePage.py, ReviewsPage.py, app.py: Streamlit Front-end files  
7. packages.txt, requirements.txt : Packages/libraries required to run the app  

# Front-end
Used Streamlit to deploy the app

<h3>HomePage</h3>

![](Images/homepage.png)

<h3>ResultsPage</h3>

![](Images/showresults1.png)

![](Images/showresults2.png)


# Data Preparation
Converted text into lowercase.  
Changed negation statements into positive form to account for context.  
Changed words to their base form.  
Converted words into the singular form.  
Removed any stopwords and long words(length>15).  

# Feature Selection
Pointwise Mutual Information(PMI) on words to identify the importance of words, bigrams, trigrams to the category.  
Selected top n features with the highest PMI scores to build the vocabulary

# Text Vectorization
Applied Tf-idf Vectorizer on pre-processed reviews using the vocabulary built using PMI feature selection.  

# Model
Used an SVC model with rbf kernel and a few hyperparameters tuning to get the ML model







