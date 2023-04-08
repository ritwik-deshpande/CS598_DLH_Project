import numpy as np

from data_preprocessing import DataPreprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 600


def tf_idf_matrix_gen(df, disease_name):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

    docs = df['text'].values
    tfidf_matrix = vectorizer.fit_transform(docs)

    words = vectorizer.get_feature_names_out()

    X = tfidf_matrix.toarray()
    Y = np.array(df[disease_name].values)
    print(words, X.shape, Y.shape)
    return X, Y, words

def main():
    dataPreprocessing = DataPreprocessing('../train/train_data_textual.csv', 'Asthma')
    preprocessed_df = dataPreprocessing.preprocess_data()

    X, Y, words = tf_idf_matrix_gen(preprocessed_df, 'Asthma')

    print(X.shape, Y.shape, words)



if __name__ =='__main__':
    main()