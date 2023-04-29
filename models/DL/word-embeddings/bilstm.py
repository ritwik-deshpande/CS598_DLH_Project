import csv
import sys
import os
import pandas as pd
import argparse
module_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.append(module_path)
from dataset.preprocessing.word2vec_embeddings_gen import Word2VecFeatureGeneration
os.chdir(module_path)

def train_and_validate(hidden_size_1, hidden_size_2, n_splits, epochs, X, Y):
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score
   
    class BiLSTM(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers, output_size):
            super(BiLSTM, self).__init__()
            self.hidden_size_1 = hidden_size_1
            self.hidden_size_2 = hidden_size_2
            self.num_layers = num_layers
            self.bilstm1 = nn.LSTM(input_size, hidden_size_1, num_layers, batch_first=True, bidirectional=True)
            self.bilstm2 = nn.LSTM(hidden_size_1*2, hidden_size_2, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2*hidden_size_2, output_size)

        def forward(self, x):
            h0_1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size_1, dtype=torch.float32).to(x.device)
            c0_1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size_1, dtype=torch.float32).to(x.device)
            out, _ = self.bilstm1(x, (h0_1, c0_1))
            # print(out.shape)
            h0_2 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size_2, dtype=torch.float32).to(x.device)
            c0_2 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size_2, dtype=torch.float32).to(x.device)
            out, _ = self.bilstm2(out, (h0_2, c0_2))
            # print(out.shape)
            out = self.fc(out[:, -1, :])
            return out
    
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

        criterion = nn.MSELoss()
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


        f1_macro = f1_score(Y_val_fold.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        f1_micro = f1_score(Y_val_fold.cpu().numpy(), y_pred.cpu().numpy(), average='micro')
        # print(f"The f1 macro score is {f1_macro} and f1_micro score is {f1_micro}")

        f1_macro_list.append(f1_macro)
        f1_micro_list.append(f1_micro)

    f1_macro = np.mean(f1_macro_list)
    f1_micro = np.mean(f1_micro_list)
    print(f"The f1 macro score is {f1_macro} and f1_micro score is {f1_micro}")

    return f1_macro, f1_micro


def main(hidden_size_1, hidden_size_2, n_splits, epochs):
    all_f1_macro_scores = []
    all_f1_micro_scores = []
    morbidities = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypercholesterolemia', 'Hypertension', 'Hypertriglyceridemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous-Insufficiency']

    column_headings = ["Morbidity Class", "DL_Macro F1", "DL_Micro F1"]

    with open("./results/word-embeddings-features/performance_DL.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([column_headings[0], column_headings[1], column_headings[2]])
    
    for morbidity in morbidities[:1]:
        train_preprocessed_df = pd.read_csv('./dataset/train/train_data_intuitive_preprocessed.csv')
        train_preprocessed_df = train_preprocessed_df[train_preprocessed_df[morbidity].isin([1.0, 0.0])]

        X, Y, words = Word2VecFeatureGeneration(train_preprocessed_df, morbidity).word2vec_matrix_gen()
    
        f1_macro, f1_micro = train_and_validate(hidden_size_1, hidden_size_2, n_splits, epochs, X, Y)
        data = [f1_macro, f1_micro]
        all_f1_macro_scores.append(f1_macro)
        all_f1_micro_scores.append(f1_micro)

        row_heading = morbidity
        with open("./results/word-embeddings-features/performance_DL.csv", "a", newline="") as file:
            writer = csv.writer(file)
            row = [row_heading]
            row.extend(data)
            writer.writerow(row)

    with open("./results/word-embeddings-features/performance_DL.csv", "a", newline="") as file:
        writer = csv.writer(file)
        row = ["Overall-Average"]
        row.extend([sum(all_f1_macro_scores)/len(all_f1_macro_scores),  sum(all_f1_micro_scores)/len(all_f1_micro_scores) ])
        writer.writerow(row)

if __name__ == '__main__':
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