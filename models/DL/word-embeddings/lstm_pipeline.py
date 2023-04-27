
def train():
    import torch
    import torch.nn as nn
    import pandas as pd
    import sys
    import os
    module_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    sys.path.append(module_path)
    from dataset.preprocessing.word2vec_embeddings_gen import Word2VecFeatureGeneration
    os.chdir(module_path)



    class BiLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(BiLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bilstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.bilstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2*hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size, dtype=torch.float32).to(x.device)
            out, _ = self.bilstm1(x, (h0, c0))
            print(out.shape)
            out, _ = self.bilstm2(out, (h0, c0))
            print(out.shape)
            out = self.fc(out[:, -1, :])
            return out

    morbidity = 'Asthma'
    
    train_preprocessed_df = pd.read_csv('./dataset/train/train_data_intuitive_preprocessed.csv')
    train_preprocessed_df = train_preprocessed_df[train_preprocessed_df[morbidity].isin([1.0, 0.0])]

    X, Y, words = Word2VecFeatureGeneration(train_preprocessed_df, morbidity).word2vec_matrix_gen()
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape)


    #Model parameters
    input_size = 100
    hidden_size = 10
    num_layers = 2
    output_size = 1

    bilstm = BiLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(bilstm.parameters(), lr=0.01)


    # Train model
    for epoch in range(3):
        # Forward pass
        outputs = bilstm(X)
        print(outputs.shape, Y.unsqueeze(1).shape)
        loss = criterion(outputs, Y.unsqueeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} loss is {loss}")

    torch.save(bilstm.state_dict(), f'./models/DL/word-embeddings/model_{morbidity}.pth')


if __name__ == '__main__':
    train()