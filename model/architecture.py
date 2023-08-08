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

class SMCNN(nn.Module):
    def __init__(self, dropout_rate=0.001):
        super(SMCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 25)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x