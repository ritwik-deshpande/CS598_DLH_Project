import pandas as pd
import spacy


class DataPreprocessing:
    def __init__(self):
        self.nlp  = spacy.load("en_core_web_sm")        

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

    def one_hot_encoding(self, row):
        for row_idx in range(2, len(row)):
            if row[row_idx] == 'Y':
                row[row_idx] = 1.0
            elif row[row_idx] == 'N':
                row[row_idx] = 0.0
            else:
                row[row_idx] = -1
        return row

    def preprocess_data(self, inp_csv_file, out_csv_file):
        self.df = pd.read_csv(inp_csv_file)
        self.df = self.df.apply(self.to_lower_case, axis=1)
        self.df = self.df.apply(self.tokenize, axis=1)
        self.df = self.df.apply(self.remove_punctuation_and_numeric_values, axis=1)
        self.df = self.df.apply(self.lemmatization, axis=1)
        self.df = self.df.apply(self.join, axis=1)
        self.df = self.df.apply(self.one_hot_encoding, axis=1)
        self.df.to_csv(out_csv_file, index=False)

        # print(self.df)



