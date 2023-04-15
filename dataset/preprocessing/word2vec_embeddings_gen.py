import numpy as np
from gensim.models import Word2Vec
import collections

VECTOR_SIZE = 100


class Word2VecFeatureGeneration:
    def __init__(self, df, disease_name):
        self.df = df
        self.disease_name = disease_name
    def word2vec_matrix_gen(self):
        sentences = self.df['text'].apply(lambda x: x.split(' ')).values
        # print(sentences)
        model = Word2Vec(sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4)

        X = []

        for i, sentence in enumerate(sentences):
            word_vectors = []
            for j, word in enumerate(sentence):
                word_vectors.append(model.wv.get_vector(word))
            X.append(word_vectors)
        X = np.array(X, dtype=list)
        Y = np.array(self.df[self.disease_name].values)
        words = model.wv.key_to_index.keys()
        print(X.shape, Y.shape, collections.Counter(list(Y)))

        return X, Y, words

