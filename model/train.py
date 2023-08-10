from model.architecture import MnistClassifier
from skorch import NeuralNetClassifier
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import ParameterGrid
from model.architecture import *
from torch.utils.data import Dataset, DataLoader

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

def train_model_CNN(data, model, n_epochs=20, lr=0.001):
    optimizer = Adam(model.parameters(), lr=0.001)    
    L = nn.CrossEntropyLoss()
    
    losses = []
    epochs = []
    
    print(f"Training a Convolutional Neural Network for {n_epochs} epochs,"
          f"with learning rate {lr}")
    
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

# Grid Search Function
def grid_search(train_data, test_data, num_epochs=20):
    param_grid = {
        'dropout_rate': list(np.arange(1e-7,1e-5,((1e-5-1e-7)/3))),
        'weight_decay': list(np.arange(1e-7,1e-5,((1e-5-1e-7)/3))),
        'learning_rate': list(np.arange(1e-5,1e-2,((1e-2-1e-5)/3)))
                             }
    
    train_dl = DataLoader(train_data, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=32)
    
    best_accuracy = 0.0
    best_params = None

    for params in ParameterGrid(param_grid):
        model_cnn = SMCNN_gs(dropout_rate=params['dropout_rate'], weight_decay=params['weight_decay'])
        optimizer = Adam(model_cnn.parameters(), lr=params['learning_rate'])
        epoch_data, loss_data = train_model_CNN(train_dl, model_cnn, num_epochs)
        _, test_accuracy = evaluate_mode(train_dl, test_dl, model_cnn)
        print("Hyperparameters:", params)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_params = params
    
    return best_accuracy, best_params