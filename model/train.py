from model.architecture import MnistClassifier
from skorch import NeuralNetClassifier
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

def train_model_fc(X_train: np.array, y_train: np.array,
                   learning_rate=0.02,
                   epochs=100,
                   ):
    torch.manual_seed(0)
    print(f"Training a fully connected neural network for {epochs} epochs,"
          f"with learning rate {learning_rate}")
    net = NeuralNetClassifier(MnistClassifier,
                              max_epochs=epochs,
                              lr=learning_rate
                              )
    net.fit(X_train, y_train)
    return net

def train_model_CNN(data, model, n_epochs=20):
    optimizer = Adam(model.parameters(), lr=0.001)    
    L = nn.CrossEntropyLoss()
    
    losses = []
    epochs = []
    
    for epoch in range(n_epochs):
        print(f"Training epoch {epoch}.")
        for i, (x, y) in enumerate(data):
            optimizer.zero_grad()
            outputs = model(x)
            loss_value = L(outputs, y)
            loss_value.backward()
            optimizer.step()
            epochs.append(epoch + i / len(data))
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)