import numpy as np
import collections
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_FEATURES = 600

class TFIDFFeatureGeneration:
    def __init__(self, df, disease_name):
        self.df = df
        self.disease_name = disease_name

    def tf_idf_matrix_gen(self, ):
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

        docs = self.df['text'].values
        tfidf_matrix = vectorizer.fit_transform(docs)

        words = vectorizer.get_feature_names_out()

        X = tfidf_matrix.toarray()
        Y = np.array(self.df[self.disease_name].values)
        print(X.shape, Y.shape, collections.Counter(list(Y)))
        return X, Y, words
