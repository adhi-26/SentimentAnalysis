import pandas as pd
import numpy as np
import pickle as pkl
from id_generator import serial_id_generator
from pandarallel import pandarallel
import os
import datetime

def generate_ngram(n, min_df, df: pd.DataFrame, folder: str, directory_path):

    prefix_dict = {1:'uni',
                   2:'bi',
                   3:'tri',
                   4:'quad',
                   5:'pent',
                   6:'hex'
                   }
    
    def ngrams(code_list: list, ngram=n):
        return [' '.join([code_list[i+idx] for i in range(ngram)]) for idx,_ in enumerate(code_list[:-n+1])]

    # create folder to save files
    directory_path = 'pickles/PMI_scores/' + folder

    # create id to distinguish files
    id = f'{prefix_dict[n]}_mindf{min_df}_' + folder 

    # list of words in the text (Series)
    df['word_list'] = df['Processed_Reviews'].str.split().parallel_apply(ngrams)
  
    # count of various ngrams in the document (Series) (Index = ngrams) (Values = count)
    ngram_count = df['word_list'].explode().value_counts()
    # setting threshhold for ngram frequency (Series) (Index = ngrams) (Values = count)
    ngram_set = ngram_count[ngram_count>min_df]
    # ngram series to iterate over
    ngram_corpus = pd.Series(ngram_set.index)
    print(f'Starting {n}-gram PMI score calculation')
    pmi_scores = ngram_corpus.parallel_apply(lambda x: pmi(x, df, min_df = min_df))
    print('ngram PMI calculation done')
    
    pmi_scores_df = pd.DataFrame(pmi_scores.to_list(), index=ngram_corpus.values)
    pmi_scores_df.sort_values(by='PMI', ascending= False, inplace=True)
    date = str(datetime.datetime.now())
 
    # pickling pmi_scores  
    pkl.dump(pmi_scores_df, open(f'{directory_path}/{id}.pkl', 'wb'))

    # writing pmi_score details in a text file
    with open(f'{directory_path}/{id}_details.txt', 'a') as f:
        f.write('Details\n')
        f.write(f'ngram type: {prefix_dict[n]}gram\n')
        f.write(f'Date of creation: {date}\n')
        f.write(f'Minimum document frequency: {min_df}')

    return pmi_scores_df

def get_word_set(data: pd.DataFrame, min_count=20):
    word_list = data['Processed_Reviews'].str.split().explode()
    count = word_list.value_counts()
    word_set = count[count>=min_count]
    return word_set

def pmi(word, data, min_df = 20):

    # 1. number of rows that contains word
    def find_word(row, word):
        return 1 if word in row['word_list'] else 0
    
    # 2. number of rows where target is target and row contains pos
    def find_word_target(row, word, target):
        return 1 if row['label']==target and word in row['word_list'] else 0
    
    # 3. number of rows where target is target 
    def find_target(row, target):
        return 1 if row['label']==target else 0
    
    # 3/2
    def conditional_prob(word, target):
        pair_freq = data.apply(lambda row: find_word_target(row, word, target), axis=1).sum()
        target_freq = data.apply(lambda row: find_target(row, target), axis=1).sum()
        return pair_freq / target_freq
    
    # 1
    def prob_word(word):
        word_df = data.apply(lambda row: find_word(row, word), axis=1).sum()
        den = len(data)
        return word_df/den, word_df
    
    word_prob, word_df = prob_word(word)

    pos_score = conditional_prob(word, 'pos')/word_prob if word_df>=min_df else 1
    neg_score = conditional_prob(word, 'neg')/word_prob if word_df>=min_df else 1

    return {'PMI' : np.max([np.log(pos_score), np.log(neg_score)]), 'doc_freq' : word_df}
    # return {'PMI' : np.max([np.log(pos_score), np.log(neg_score)])}

def unigram(min_df, df: pd.DataFrame, folder: str, directory_path):
        
    # create folder to save files
    directory_path = 'pickles/PMI_scores/' + folder

    #create id to distinguish files
    id = f'uni_mindf{min_df}_' + folder  

    df['word_list'] = df['Processed_Reviews'].str.split()
    word_list = df['word_list'].explode()
    # unique words and their counts
    count = word_list.value_counts()
    #filtering by minimum threshold
    word_set = count[count>=min_df]
    # words as series
    corpus_series = pd.Series(word_set.index)

    print('Starting PMI Scores calculation')
    pmi_scores = corpus_series.parallel_apply(lambda x: pmi(x,df, min_df))
    print('PMI scores calculated')
    # PMI Scores as dataframe
    pmi_scores_df = pd.DataFrame(pmi_scores.to_list(), index=corpus_series.values)
    pmi_scores_df.sort_values(by='PMI', ascending=False, inplace=True)
    date = str(datetime.datetime.now())

    # pickling pmi_scores  
    pkl.dump(pmi_scores_df, open(f'{directory_path}/{id}.pkl', 'wb'))

    # writing pmi_score details in a text file
    with open(f'{directory_path}/{id}_details.txt', 'a') as f:
        f.write('Details\n')
        f.write(f'ngram type: unigram\n')
        f.write(f'Date of creation: {date}\n')
        f.write(f'Minimum document frequency: {min_df}')

    return pmi_scores_df

if __name__ == '__main__':
    pandarallel.initialize(nb_workers=os.cpu_count()-1, progress_bar=True)
    data = pd.read_csv('Data/processed_training_data.csv', index_col=None)

    #creating folder
    folder_name = serial_id_generator()
    directory_path = 'pickles/PMI_scores/' + folder_name
    os.mkdir(directory_path)

    unigram_pmi_scores = unigram(20, data.copy(), folder = folder_name, directory_path=directory_path)
    bigram_pmi_scores = generate_ngram(2, 20, data.copy(), folder = folder_name, directory_path=directory_path)
    trigram_pmi_scores = generate_ngram(3, 20, data.copy(), folder = folder_name, directory_path=directory_path)
