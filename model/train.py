from model.architecture import MnistClassifier
from skorch import NeuralNetClassifier
import torch
import numpy as np


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
