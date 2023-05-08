import csv
import argparse
from funcx import FuncXExecutor
import os
os.chdir('/Users/renalkakhan/Documents/GitHub/CS598_DLH_Project/')
def hw():
    return 'hello', 'world'

def train_and_validate(hidden_size_1, hidden_size_2, n_splits, epochs, morbidity, word_embedding):
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score
    import os
    import pandas as pd
    import sys
    import collections
    os.chdir('/repo')
    sys.path.append(os.getcwd())    
    from dataset.preprocessing.word2vec_embeddings_gen import Word2VecFeatureGeneration, GloVeFeatureGeneration, FastTextFeatureGeneration, USEFeatureGeneration

    vectorizor_dict = {
        "word2vec": Word2VecFeatureGeneration,
        "glove": GloVeFeatureGeneration,
        "fasttext": FastTextFeatureGeneration,
        "USE": USEFeatureGeneration
    }
   
    class BiLSTM(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers, output_size):
            super(BiLSTM, self).__init__()
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.num_layers = num_layers
            self.bilstm1 = nn.LSTM(input_size, hidden_size_1, num_layers, batch_first=True, bidirectional=False)
            self.bilstm2 = nn.LSTM(hidden_size_1, hidden_size_2, num_layers, batch_first=True, bidirectional=False)
            self.fc = nn.Linear(hidden_size_2, output_size)

        def forward(self, x):
            h0_1 = torch.zeros(1, x.size(0), self.hidden_size_1, dtype=torch.float32).to(x.device)
            c0_1 = torch.zeros(1, x.size(0), self.hidden_size_1, dtype=torch.float32).to(x.device)
            out, _ = self.bilstm1(x, (h0_1, c0_1))
            # print(out.shape)
            h0_2 = torch.zeros(1, x.size(0), self.hidden_size_2, dtype=torch.float32).to(x.device)
            c0_2 = torch.zeros(1, x.size(0), self.hidden_size_2, dtype=torch.float32).to(x.device)
            out, _ = self.bilstm2(out, (h0_2, c0_2))
            # print(out.shape)
            out = self.fc(out[:, -1, :])
            return out
    
    train_preprocessed_df = pd.read_csv('./dataset/train/train_data_intuitive_preprocessed.csv')
    train_preprocessed_df = train_preprocessed_df[train_preprocessed_df[morbidity].isin([1.0, 0.0])]

    X, Y, words = vectorizor_dict[word_embedding](train_preprocessed_df, morbidity).matrix_gen()
        
    

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    # print(X.shape)


    #Model parameters
    input_size = X.size(2)
    hidden_size_1 = hidden_size_1 #Following the paper
    hidden_size_2 = hidden_size_2 #Following the paper
    num_layers = 1
    output_size = 1

    f1_macro_list = []
    f1_micro_list = []

    y_pred_all_folds = []
    n_split = n_splits #Mentioned in the paper
    # Train model    
    skf = KFold(n_splits=n_split, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, Y):
        X_train_fold, Y_train_fold = X[train_idx], Y[train_idx]
        X_val_fold, Y_val_fold = X[val_idx], Y[val_idx]
        bilstm = BiLSTM(input_size, hidden_size_1, hidden_size_2, num_layers, output_size)


        if torch.cuda.is_available():
            X_train_fold = X_train_fold.cuda()
            Y_train_fold = Y_train_fold.cuda()
            X_val_fold = X_val_fold.cuda()
            Y_val_fold = Y_val_fold.cuda()
            bilstm = bilstm.cuda()
            bilstm = torch.nn.DataParallel(bilstm)

        class_counter = collections.Counter(np.array(Y_train_fold.cpu()).tolist())

        if (0.0 not in class_counter.keys()) or (1.0 not in class_counter.keys()):
            f1_micro = 1
            f1_macro = 1
        else:
            # weight  = class_counter[0.0]/class_counter[1.0]
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(bilstm.parameters(), lr=0.01)
            bilstm.train()
            for epoch in range(epochs):
                # Forward pass
                outputs = bilstm(X_train_fold)
                # print(outputs.shape, Y.unsqueeze(1).shape)
                loss = criterion(outputs, Y_train_fold.unsqueeze(1))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch} loss is {loss}")

            bilstm.eval()
            with torch.no_grad():
                y_hat = bilstm(X_val_fold)
            y_hat = y_hat.view(y_hat.shape[0])

            # print(Y_val_fold.shape, y_hat.shape)

            y_pred = []
            for val in y_hat.data:
                if val <= 0.6:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
                    
            y_pred = torch.tensor(y_pred)
            y_pred_all_folds.append(y_pred)

            f1_macro = f1_score(Y_val_fold.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            f1_micro = f1_score(Y_val_fold.cpu().numpy(), y_pred.cpu().numpy(), average='micro')
        # print(f"The f1 macro score is {f1_macro} and f1_micro score is {f1_micro}")

        f1_macro_list.append(f1_macro)
        f1_micro_list.append(f1_micro)
        filename = f'bilstm_model_{morbidity}_{word_embedding}.pt'
        #torch.save(bilstm, filename)

    f1_macro = np.mean(f1_macro_list)
    f1_micro = np.mean(f1_micro_list)
    print(f"The f1 macro score is {f1_macro} and f1_micro score is {f1_micro}")

    return y_pred_all_folds, f1_macro, f1_micro


def main(hidden_size_1, hidden_size_2, n_splits, epochs):

    morbidities = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous-Insufficiency']

    column_headings = ["Morbidity Class", "DL_Macro_F1_word2vec", "DL_Micro_F1_word2vec"\
                    , "DL_Macro_F1_glove", "DL_Micro_F1_glove"\
                    ,"DL_Macro_F1_fasttext", "DL_Micro_F1_fasttext"\
                    ,"DL_Macro_F1_USE", "DL_Micro_F1_USE"]
    
    word_embeddings = ["word2vec", "glove", "fasttext", "USE"]
    
    with open("./results/word-embeddings-features/performance_DL2.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_headings)
    
    with FuncXExecutor(endpoint_id=ENDPOINT_ID, container_id=CONTAINER_ID) as ex:
        for morbidity in morbidities:
            row_heading = morbidity
            row = [row_heading]
            for word_embedding in word_embeddings:
                print(morbidity, word_embedding)
                # fut = ex.submit(hw)
                fut = ex.submit(train_and_validate, hidden_size_1, hidden_size_2, n_splits, epochs, morbidity, word_embedding)
                res = fut.result()
                #print(res)
                y_pred_all_folds, f1_macro, f1_micro = res
                data = [f1_macro, f1_micro]
                row.extend(data)
            with open(f"./results/word-embeddings-features/y_preds_{morbidity}_{word_embedding}.txt", "w") as file0:
                    file0.write(str(y_pred_all_folds))
            with open("./results/word-embeddings-features/performance_DL2.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)
            


if __name__ == '__main__':
    CONTAINER_ID = "db367450-ab9d-421b-a296-e29d757943e5"
    ENDPOINT_ID = "8056e8e8-2e32-4c75-88f1-81ff7a932f00"
    parser = argparse.ArgumentParser(description='Train and Validate DL model')

    parser.add_argument('--epochs', '-e', type=str, help='Number of epochs')
    parser.add_argument('--n_splits', '-ns', type=str, help='Number of splits for K fold validation')
    parser.add_argument('--hidden_size_1', '-hs1', type=str, help='Hidden size1 of Biltsm layer 1')
    parser.add_argument('--hidden_size_2', '-hs2', type=str, help='Hidden size2 of Biltsm layer 2')

    # Parse the arguments
    args = parser.parse_args()
    hidden_size_1 = 20
    hidden_size_2 = 10
    n_splits = 10
    epochs = 10

    # Access the arguments
    if args.epochs:
        epochs = int(args.epochs)
    if args.hidden_size_1:
        hidden_size_1 = int(args.hidden_size_1)
    if args.hidden_size_2:
        hidden_size_2 = int(args.hidden_size_2)
    if args.n_splits:
        n_splits = int(args.n_splits)

    main(hidden_size_1, hidden_size_2, n_splits, epochs)

# CUDA_VISIBLE_DEVICES=0 python models/DL/word-embeddings/bilstm.py --epochs=20 --hidden_size_1=128 --hidden_size_2=64 --n_splits=10

# srun --account=bbmi-delta-gpu --partition=gpuA100x4-interactive --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=16 --cpus-per-task=8 --mem=64g --pty bash