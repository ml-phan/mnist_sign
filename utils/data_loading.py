from typing import Tuple

import numpy as np
import sklearn
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def get_dataset_openml() -> sklearn.utils.Bunch:
    """
    Get the dataset from openml
    :return: sklearn Bunch data object
    """
    mnist = fetch_openml('SignMNIST', as_frame=False, cache=False)
    return mnist


def data_preprocessing(mnist: sklearn.utils.Bunch) -> Tuple[np.array,
                                                            np.array]:
    """
    Extract input and target data from a sklearn Bunch data object
    :param mnist:
    :return: X and y as numpy array
    """
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X /= 255
    return X, y



def plot_example(X: np.array, y: np.array):
    """Plot the first 5 images and their labels in a row."""
    for i, (img, y) in enumerate(zip(X[:5].reshape(5, 28, 28), y[:5])):
        plt.subplot(151 + i)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(y)
