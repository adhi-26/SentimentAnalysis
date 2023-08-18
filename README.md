# SentimentAnalysis
Sentiment Analysis on Movie Reviews using Streamlit to build front-end.  
App - https://lyp6g8zsgxemtvw4q2e89d.streamlit.app/

# Dataset
Dataset with 25000k reviews for training and testing each with no more than 30 reviews from a movie
https://ai.stanford.edu/~amaas/data/sentiment/

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

# Front-end
Used Streamlit to deploy the app

<h3>HomePage</h3>

![]('Images/homepage.png')





