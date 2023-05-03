
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
import tensorflow_hub as hub
import collections

VECTOR_SIZE = 300
DOCUMENT_LENGTH = 500

glove_file_path = 'glove.6B.300d.txt'

class Word2VecFeatureGeneration:
    def __init__(self, df, disease_name):
        self.df = df
        self.disease_name = disease_name

    def word2vec_matrix_gen(self):
        self.df['split_text'] = self.df['text'].apply(lambda x: x.split(' '))
        self.df = self.df[self.df.apply(lambda row: len(row['split_text']) < DOCUMENT_LENGTH, axis=1)]
        sentences = self.df['split_text'].values
        # print(sentences)
        model = Word2Vec(sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4, epochs=10)
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
        self.VECTOR_SIZE = 300

    def get_labels(self, data):
        return self.df[self.disease_name].values.tolist()

    def load_embeddings(self, embedding_path):
        word_vectors = {}
        with open(embedding_path, encoding='utf8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                word_vectors[word] = coefs
        # Add UNK token to the vocabulary with a random vector
        word_vectors['UNK'] = np.random.rand(self.VECTOR_SIZE)
        return word_vectors

    def glove_matrix_gen(self, max_length=100):
        word_vectors = self.load_embeddings(self.glove_file_path)

        self.df['split_text'] = self.df['text'].apply(lambda x: x.split(' '))
        self.df = self.df[self.df.apply(lambda row: len(row['split_text']) < DOCUMENT_LENGTH, axis=1)]
        sentences = self.df['split_text'].values
        sentences = [s[:max_length] for s in sentences]

        X = np.zeros((len(sentences), max_length, self.VECTOR_SIZE))
        for i, sentence in enumerate(sentences):
            sentence = [word if word in word_vectors else 'UNK' for word in sentence]
            sentence_vectors = [word_vectors.get(word, word_vectors['UNK']) for word in sentence]
            sentence_vectors += [np.zeros(self.VECTOR_SIZE)] * (max_length - len(sentence))
            X[i, :, :] = np.array(sentence_vectors)
        Y = np.array(self.get_labels(self.df))
        words = list(word_vectors.keys())
        return X, Y, words



class FastTextFeatureGeneration:
    def __init__(self, df, disease_name):
        self.df = df
        self.disease_name = disease_name

    def fasttext_matrix_gen(self):
        self.df['split_text'] = self.df['text'].apply(lambda x: x.split(' '))
        self.df = self.df[self.df.apply(lambda row: len(row['split_text']) < DOCUMENT_LENGTH, axis=1)]
        sentences = self.df['split_text'].values
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
    
        self.df['split_text'] = self.df['text'].apply(lambda x: x.split(' '))
        self.df = self.df[self.df.apply(lambda row: len(row['split_text']) < DOCUMENT_LENGTH, axis=1)]
        sentences = self.df['split_text'].apply(lambda x: ' '.join(x)).values  # Join list of words back into a sentence
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        sentence_embeddings = embed(sentences)

        X = np.array(sentence_embeddings)
        Y = np.array(self.df[self.disease_name].values)
        words = []

        print(X.shape, Y.shape, collections.Counter(list(Y)))

        return X, Y, words
