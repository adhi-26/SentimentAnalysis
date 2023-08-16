import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from .singularize import singularize

STOPWORDS = stopwords.words('english')

def filter_words(text, lemma = True):
    tokens = handle_negation(text)
    tokens = remove_stopwords(tokens)
    if lemma:
        tokens = lemmatize(tokens)
    tokens = [singularize(token) for token in tokens]
    tokens = long_short_words(tokens)
    return ' '.join(tokens)

def handle_negation(sentence):	
    '''
    Input: Tokenized sentence (List of words)
    Output: Tokenized sentence with negation handled (List of words)
    '''
    tokenizer = RegexpTokenizer(r'[a-z]+')
    sentence = tokenizer.tokenize(sentence)

    temp = int(0)
    for i in range(len(sentence)):
        if sentence[i-1]=='not':
            antonyms = []
            for syn in wordnet.synsets(sentence[i]):
                syns = wordnet.synsets(sentence[i])
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp>max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i-1] = ''
    while '' in sentence:
        sentence.remove('')
    return sentence

def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOPWORDS]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, wordnet.VERB) for word in tokens]

def long_short_words(tokens, max_len = 15, min_len = 3):
    tokens = [word for word in tokens if len(word)<=max_len and len(word)>=min_len]
    return tokens

def remove_brackets(text):
    return re.sub('\(.*?\)', '', text)

def decontract(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase

def preprocess_text(text: str):
    text = text.lower()
    text = remove_brackets(text)
    text = decontract(text)
    text = filter_words(text)
    return text

