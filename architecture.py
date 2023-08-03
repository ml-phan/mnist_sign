import torch.nn as nn
import torch.nn.functional as F

mnist_dim = 784
hidden_dim = int(mnist_dim / 8)
hidden_dim2 = int(hidden_dim / 2)
output_dim = 25


class MnistClassifier(nn.Module):
    def __init__(self,
                 input_dim=mnist_dim,
                 hidden_dim=hidden_dim,
                 hidden_dim2=hidden_dim2,
                 output_dim=output_dim,
                 dropout=0,
                 ):
        super(MnistClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.relu(self.hidden2(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X
