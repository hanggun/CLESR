from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import os
import pandas as pd
import numpy as np
import collections


filename = '/home/zxa/ps/open_data/ner/genia/genia_vector_iter50.txt'
labelfile = '/home/zxa/ps/open_data/ner/genia/context_label_name.txt'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
labels = open(labelfile).read().splitlines()

dt = pd.DataFrame()

words = []
for label in labels:
    words.append(label)
    for key, value in model.most_similar(label, topn=20):
        words.append(key)
words = collections.Counter(words)

label_words_vector = {}
for label in labels:
    label_words_vector[label] = {'words': [label]}
    label_words_vector[label]['vectors'] = [model[label].tolist()]
    for key, value in model.most_similar(label, topn=20):
        if words[key] == 1:
            label_words_vector[label]['words'].append(key)
            label_words_vector[label]['vectors'].append(model[key].tolist())

vectors = []
vector_length = []
for label in labels:
    vectors.extend(label_words_vector[label]['vectors'])
    vector_length.append(len(label_words_vector[label]['vectors']))

tsne_vec = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=4).fit_transform(np.array(vectors))
x, y = tsne_vec[:, 0], tsne_vec[:, 1]
color_name = ['blue', 'orange', 'green', 'red', 'purple']
word_ind_vec, label_ind_vec = [], []
colors = []
color_count = -1
for i in range(len(vectors)):
    if i in [0, vector_length[0]+1, sum(vector_length[0:2])+2, sum(vector_length[0:3])+3, sum(vector_length[0:4])+4]:
        label_ind_vec.append(i)
        color_count += 1
    else:
        word_ind_vec.append(i)
        colors.append(color_name[color_count])
plt.scatter(x[word_ind_vec], y[word_ind_vec], c=colors)
for i,j,color,l in zip(x[label_ind_vec].tolist(), y[label_ind_vec].tolist(), color_name, labels):
    plt.scatter(i, j, c=color, marker='^', s=300,
                label=l)
plt.legend()
plt.show()