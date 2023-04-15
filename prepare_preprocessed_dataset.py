from dataset.preprocessing.data_preprocessing import DataPreprocessing

dataPreprocessing = DataPreprocessing()
dataPreprocessing.preprocess_data('./dataset/train/train_data_intuitive.csv', './dataset/train/train_data_intuitive_preprocessed.csv')
dataPreprocessing.preprocess_data('./dataset/test/test_data_intuitive.csv', './dataset/test/test_data_intuitive_preprocessed.csv')
