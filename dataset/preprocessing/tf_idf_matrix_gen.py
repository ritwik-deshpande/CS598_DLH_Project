import numpy as np

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
        print(words, X.shape, Y.shape)
        return X, Y, words

# def main():
#     dataPreprocessing = DataPreprocessing('../train/train_data_textual.csv', 'Asthma')
#     preprocessed_df = dataPreprocessing.preprocess_data()
#
#     X, Y, words = TF_IDF_FeatureGeneration(preprocessed_df, 'Asthma').tf_idf_matrix_gen()
#
#     print(X.shape, Y.shape, words)


#
# if __name__ =='__main__':
#     main()