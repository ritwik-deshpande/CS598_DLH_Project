
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
import tensorflow as tf
import tensorflow_hub as hub
import collections
import os
os.chdir('/Users/renalkakhan/Documents/GitHub/CS598_DLH_Project/')

VECTOR_SIZE = 100
glove_file_path = 'glove.6B.100d.txt'


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


class GloVeFeatureGeneration:
    def __init__(self, df, disease_name):
        self.df = df
        self.disease_name = disease_name
        self.glove_file_path = glove_file_path

    def get_labels(self, data):
        return data[self.disease_name].values.tolist()

    def load_embeddings(self, embedding_path):
        word_vectors = {}
        with open(embedding_path, encoding='utf8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                word_vectors[word] = coefs
        # Add UNK token to the vocabulary with a random vector
        word_vectors['UNK'] = np.random.rand(100)
        return word_vectors
    
    
    def glove_matrix_gen(self, max_length=100):
        word_vectors = self.load_embeddings(self.glove_file_path)

        sentences = self.df['text'].apply(lambda x: x.split()).tolist()
        sentences = [s[:max_length] for s in sentences]

        X = np.zeros((len(sentences), max_length, VECTOR_SIZE))
        for i, sentence in enumerate(sentences):
            sentence = [word if word in word_vectors else 'UNK' for word in sentence]
            sentence_vectors = [word_vectors.get(word, word_vectors['UNK']) for word in sentence]
            sentence_vectors += [np.zeros(VECTOR_SIZE)] * (max_length - len(sentence))
            X[i, :, :] = np.array(sentence_vectors)
        Y = np.array(self.get_labels(self.df))
        words = list(word_vectors.keys())
        return X, Y, words


class FastTextFeatureGeneration:
    def __init__(self, df, disease_name):
        self.df = df
        self.disease_name = disease_name

    def fasttext_matrix_gen(self):
        sentences = self.df['text'].apply(lambda x: x.split(' ')).values
        fasttext_model = FastText(sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4)

        max_length = max(len(sentence) for sentence in sentences)
        X = np.zeros((len(sentences), max_length, VECTOR_SIZE)) 

        for i, sentence in enumerate(sentences):
            sentence = [word if word in fasttext_model.wv.key_to_index else 'UNK' for word in sentence]
            sentence_vectors = [fasttext_model.wv[word] for word in sentence]
            sentence_vectors += [np.zeros(VECTOR_SIZE)] * (max_length - len(sentence))
            # Add the padded sentence to the padded array
            X[i] = np.array(sentence_vectors)

        Y = np.array(self.df[self.disease_name].values)
        words = fasttext_model.wv.key_to_index.keys()
        print(X.shape, Y.shape, collections.Counter(list(Y)))

        return X, Y, words


class USEFeatureGeneration:
    def __init__(self, df, disease_name):
        self.df = df
        self.disease_name = disease_name

    def use_matrix_gen(self):
    
        sentences = self.df['text'].values
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        sentence_embeddings = embed(sentences)

        X = np.array(sentence_embeddings)
        Y = np.array(self.df[self.disease_name].values)
        words = []

        print(X.shape, Y.shape, collections.Counter(list(Y)))

        return X, Y, words
