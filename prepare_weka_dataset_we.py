import pandas as pd
import numpy as np
from dataset.preprocessing.word2vec_embeddings_gen import Word2VecFeatureGeneration
from dataset.preprocessing.pandas2arff import pandas2arff


morbidities = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous-Insufficiency']

for morbidity in morbidities:
    print(morbidity)
    train_preprocessed_df = pd.read_csv('./dataset/train/train_data_intuitive_preprocessed.csv')[['Doc_id', 'text', morbidity]]
    train_preprocessed_df = train_preprocessed_df[train_preprocessed_df[morbidity].isin([1.0, 0.0])]

    test_preprocessed_df = pd.read_csv('./dataset/test/test_data_intuitive_preprocessed.csv')[['Doc_id', 'text', morbidity]]
    test_preprocessed_df = test_preprocessed_df[test_preprocessed_df[morbidity].isin([1.0, 0.0])]

    
    X_train, Y_train, words_train = Word2VecFeatureGeneration(train_preprocessed_df, morbidity).word2vec_matrix_gen()
    X_test, Y_test, words_test = Word2VecFeatureGeneration(test_preprocessed_df, morbidity).word2vec_matrix_gen()
    
    
    X_train = np.average(X_train, axis=1)
    X_test = np.average(X_test, axis=1)

    X = np.column_stack((X_train, Y_train))
    no_of_columns = X_train.shape[1]
    columns = ['f' + str(i) for i in range(no_of_columns)] + ['class']
    pandas2arff(pd.DataFrame(X, columns=columns), f'./dataset/train/train_{morbidity}_we.arff')
    
    X = np.column_stack((X_test, Y_test))
    no_of_columns = X_test.shape[1]
    columns = ['f' + str(i) for i in range(no_of_columns)] + ['class']
    pandas2arff(pd.DataFrame(X, columns=columns), f'./dataset/test/test_{morbidity}_we.arff')