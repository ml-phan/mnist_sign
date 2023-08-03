from architecture import MnistClassifier
from skorch import NeuralNetClassifier
import torch
import numpy as np


def train_model_fc(X_train: np.array, y_train: np.array):
    torch.manual_seed(0)

    net = NeuralNetClassifier(MnistClassifier,
                              max_epochs=100,
                              lr=0.02
                              )
    net.fit(X_train, y_train)
    return net
