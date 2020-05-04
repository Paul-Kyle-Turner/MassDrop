import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, MDS

if __name__ == '__main__':

    appearance_df = pd.read_csv('Android-API/drebin-215-dataset-5560malware-9476-benign.csv')
    class_labels = appearance_df.loc[:, 'class'].tolist()

    appearance = appearance_df.loc[:, 'transact':'WRITE_SECURE_SETTINGS'].to_numpy()

    for i in range(appearance.shape[0]):
        for j in range(appearance.shape[1]):
            if type(appearance[i][j]) != int:
                if appearance[i][j] == '0':
                    appearance[i][j] = 0
                else:
                    appearance[i][j] = 1

    print(appearance.astype(float))
    appearance = appearance.astype(float)
    print(appearance.shape)

    perplexity_array = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    max_iter = 10000
    random_state = 44
    embeddings = []
    PCA = False
    PCA_to = 50
    for perp in perplexity_array:
        tsne = TSNE(perplexity=perp, n_iter=max_iter, random_state=random_state)
        xy_coordinates = tsne.fit_transform(appearance, y=class_labels)
        embeddings.append(xy_coordinates)

    with open('Android-API/embeddings_iter_10000_stop_300.pickle', 'wb') as out_embeddings:
        pickle.dump(embeddings, out_embeddings)