import pandas as pd
import spacy


class DataPreprocessing:
    def __init__(self, csv_file, disease_name):
        self.disease_name = disease_name
        self.nlp  = spacy.load("en_core_web_sm")
        self.df = pd.read_csv(csv_file)[['Doc_id', 'text', disease_name]]
        self.df = self.df[self.df[disease_name].isin(['Y', 'N'])]
        print("The shape of df is", self.df.shape)

    def to_lower_case(self, row):
        row[1] = str.lower(row[1])
        return row

    def tokenize(self, row):
        row[1] = self.nlp(row[1])
        return row


    def remove_punctuation_and_numeric_values(self, row):
        row[1] = [token for token in row[1] if token.is_alpha]
        return row


    def lemmatization(self, row):
        row[1] = [token.lemma_ for token in row[1]]
        return row

    def join(self, row):
        row[1] = ' '.join([token for token in row[1]])
        return row


    def preprocess_data(self):
        self.df = self.df.apply(self.to_lower_case, axis=1)
        self.df = self.df.apply(self.tokenize, axis=1)
        self.df = self.df.apply(self.remove_punctuation_and_numeric_values, axis=1)
        self.df = self.df.apply(self.lemmatization, axis=1)
        self.df = self.df.apply(self.join, axis=1)
        return self.df
        # print(self.df)
def main():
    dataPreprocessing = DataPreprocessing('../train/train_data_intuitive.csv', 'Asthma')
    dataPreprocessing.preprocess_data()



if __name__ =='__main__':
    main()


