import os
import pandas as pd
import datetime
import time
import pickle
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from id_generator import serial_id_generator
import numpy as np
import pandarallel
from preprocessing_text import preprocess_text
from build_vocabulary import generate_ngram, unigram
from id_generator import serial_id_generator

def load_data(directory = 'Datasets', limit: int = 0):
    pandarallel.initialize(nb_workers=os.cpu_count()-1, progress_bar=True)
    for file in os.listdir(directory):
        filename = f"{directory}/{file}"
        if 'training_data.csv' in file:
            data_train = pd.read_csv(filename)
            data_train['Processed_Reviews'] = data_train['Reviews'].parallel_apply(preprocess_text)
            # save processed training data in a csv file
            data_train.to_csv('ProcessedData/processed_training_data.csv', index=None)
            data_train = data_train.sample(frac=1)
        if 'testing_data.csv' in file:
            data_test = pd.read_csv(filename)
            data_test['Processed_Reviews'] = data_test['Reviews'].parallel_apply(preprocess_text)
            # save processed test data in a csv file
            data_train.to_csv('ProcessedData/processed_testing_data.csv', index=None)
            data_test = data_test.sample(frac=1)
    if limit:
        data_train, data_test = data_train[0:limit], data_test[0:int(limit/4)]
    return data_train[0:], data_test[0:int(len(data_train)/4)]

def feature_selection():

    #Building Vocabulary
    data = pd.read_csv('Datasets/ProcessedData/processed_training_data.csv', index_col=None)
    folder = serial_id_generator()
    path = 'pickles/PMI_scores/' + folder
    os.mkdir(path)
    
    # Minimum doc frequency of 30
    unigram_scores = unigram(30, data.copy(), folder, path)

    # Minimum doc frequency of 30
    bigram_scores = generate_ngram(2, 30, data.copy(), folder, path)

    # Minimum doc frequency of 20
    trigram_scores = generate_ngram(2, 20, data.copy(), folder, path)

    pmi_scores = pd.concat((unigram_scores, bigram_scores, trigram_scores))

    #Selecting only features in the top 50% quantile
    quantile = 0.50
    vocab_dict = {word: idx for idx,word in enumerate(pmi_scores[pmi_scores.PMI.values>np.quantile(pmi_scores.PMI,quantile)].index)}
    return vocab_dict, {'feature selection': f'top {quantile*100}% quantile of pmi scores',
                        'ngram': ['unigram','bigram', 'trigram'],
                        'min_df': [30, 30, 20]}

def svm_model(data_train, data_test):
    ngram = (1,3)
    vocabulary, feature_selection_details = feature_selection()
    print('processing done\nstarting training')
    #vectorizing data:
    vectorizer = TfidfVectorizer(ngram_range= ngram,
                                sublinear_tf=True,
                                vocabulary=vocabulary,
                                use_idf=True,
                                smooth_idf=True
                                )

    train_vectors = vectorizer.fit_transform(data_train['Processed_Reviews'])
    test_vectors = vectorizer.transform(data_test['Processed_Reviews'])
    print('vectorizer created')

    #creating a svc model:
    classifier_linear = svm.SVC(kernel='rbf')
    t0 = time.time()
    classifier_linear.fit(train_vectors, data_train['label'])
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()

    time_linear_train = t1-t0
    time_linear_predict = t2-t1

    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    report = classification_report(data_test['label'], prediction_linear, output_dict=True)
    print('positive: ', report['pos'])
    print('negative: ', report['neg'])
    time_report = {'training': time_linear_train, 'prediction': time_linear_predict}
    return vectorizer, classifier_linear, report, time_report, feature_selection_details

if __name__ == '__main__':
    pandarallel.initialize(nb_workers=os.cpu_count()-1, progress_bar=True)
    model_type = 'SVC'
    train, test = load_data()
    print(len(train), len(test))
    vectorizer, model, report, time_report, feature_selection_details = svm_model(train, test)
    model_id = serial_id_generator(model_type)
    os.mkdir(f'pickles/{model_id}')
    pickle.dump(vectorizer, open(f'pickles/{model_id}/{model_id}_vectorizer.sav', 'wb'))
    pickle.dump(model, open(f'pickles/{model_id}/{model_id}_model.sav', 'wb'))
    date = str(datetime.datetime.now())

    with open(f'pickles/{model_id}/{model_id}_Performance.txt','a') as f:
        f.write(f"{model_id}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f'Kernel: {model.kernel}\n')
        f.write(f"Time of Creation: {date}\n\n")
        f.write(f"Training Size: {len(train)}\n")
        f.write(f"Testing Size: {len(test)}\n")
        f.write("\nTraining time: %fs; Prediction time: %fs" % (time_report['training'], time_report['prediction']))
        f.write(f"\nPositive: {report['pos']}")
        f.write(f"\nNegative: {report['neg']}")
        f.write('\n\nVECTORIZER\n')
        f.write(f'\nFeature Selection: {feature_selection_details}\n')
        f.write("Parameters:\n")
        for item in vectorizer.get_params().items():
            f.write(f"\t{item}\n")