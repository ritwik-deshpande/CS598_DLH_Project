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
        max_length = max(len(sentence) for sentence in sentences)

        X = np.zeros((len(sentences), max_length, VECTOR_SIZE)) 

        for i, sentence in enumerate(sentences):
            sentence = [word if word in model.wv.key_to_index else 'UNK' for word in sentence]
            sentence_vectors = [model.wv[word] for word in sentence]
            sentence_vectors += [np.zeros(VECTOR_SIZE)] * (max_length - len(sentence))
            # Add the padded sentence to the padded array
            X[i] = np.array(sentence_vectors)
        
        Y = np.array(self.df[self.disease_name].values)
        words = model.wv.key_to_index.keys()
        print(X.shape, Y.shape, collections.Counter(list(Y)))

        return X, Y, words

