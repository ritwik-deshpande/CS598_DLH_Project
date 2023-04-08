import numpy as np

from data_preprocessing import DataPreprocessing
from gensim.models import Word2Vec

VECTOR_SIZE = 10

def word2vec_matrix_gen(df, disease_name):
    sentences = df['text'].apply(lambda x: x.split(' ')).values
    # print(sentences)
    model = Word2Vec(sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4)

    X = []

    for i, sentence in enumerate(sentences):
        word_vectors = []
        for j, word in enumerate(sentence):
            word_vectors.append(model.wv.get_vector(word))
        X.append(word_vectors)
    X = np.array(X)
    Y = np.array(df[disease_name].values)
    words = model.wv.key_to_index.keys()
    return X, Y, words

def main():
    dataPreprocessing = DataPreprocessing('../train/train_data_textual.csv', 'Asthma')
    preprocessed_df = dataPreprocessing.preprocess_data()
    print("Completed preprocessing")
    X, Y, words = word2vec_matrix_gen(preprocessed_df, 'Asthma')

    print(X.shape, Y.shape, words)


if __name__ =='__main__':
    main()