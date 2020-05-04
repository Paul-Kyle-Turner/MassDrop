import mifs
import pandas as pd
import numpy as np
import os
from joblib import load, dump
import pickle


if __name__ == '__main__':

    os.chdir('Malware-API')

    tfidf_vectors = load('tfidf_vectors.joblib').todense()
    tfidf_vectors_pd = pd.DataFrame(tfidf_vectors).values
    #print(type(tfidf_vectors_pd))
    #print(tfidf_vectors_pd)

    labels = pd.read_csv('labels.csv', header=None).values
    #print(type(labels))
    #print(labels.head())

    meta_data = pd.read_csv('meta_data_malwareAPI_10gram.csv')
    #print(type(meta_data))
    #print(meta_data.head())

    print(len(meta_data['Api-Call']))
    print(tfidf_vectors_pd.shape)
    print(labels.shape)

    tfidf_vectors_pd_col = meta_data['Api-Call']

    label_set = []
    for label in labels:
        if label[0] not in label_set:
            label_set.append(label[0])
    print(label_set)

    temp_labels = []
    for label in labels:
        for i_label in range(len(label_set)):
            if label[0] is label_set[i_label]:
                temp_labels.append(i_label)
    print(temp_labels)

    feature_selctor = mifs.MutualInformationFeatureSelector()
    feature_selctor.fit(tfidf_vectors_pd, temp_labels)

    save_list = feature_selctor.support_
    rank_list = feature_selctor.ranking_

    dump(save_list, 'saved_features.joblib')
    dump(rank_list, 'saved_features.joblib')

    print(save_list)
    print(rank_list)

