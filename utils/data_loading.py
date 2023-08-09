from typing import Tuple
import seaborn as sns
import numpy as np
import sklearn
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import torch
import pandas as pd
from sklearn.utils import Bunch
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_dataset_openml() -> sklearn.utils.Bunch:
    """
    Get the dataset from openml
    :return: sklearn Bunch data object
    """
    print(f"Loading data set SignMNIST from openml...")
    mnist = fetch_openml('SignMNIST', as_frame=False, cache=False)
    print(f"Finished loading")
    return mnist


def data_preprocessing(mnist: sklearn.utils.Bunch) -> Tuple[np.array, np.array,
                                                            np.array, np.array]:
    """
    Extract input and target data from a sklearn Bunch data object
    :param mnist:
    :return: X and y as numpy array
    """
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X /= 255
    test_split = 0.2
    print(f"Splitting the data to {int((1-test_split)*100)}% training data and "
          f" {int(test_split*100)}% test data, with"
          f" random state = {42}")
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_split,
                                                        random_state=42)
    print(f"Training data size: {len(X_train)}")
    print(f"Test data size: {len(X_test)}")
    print("Displaying classes distribution. Close the figure windows to"
          "continue...")
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("dark")
    sns.histplot(y_train, bins=25, ax=axs[0])
    axs[0].set_title("Distribution of training set classes")
    sns.histplot(y_test, bins=25, ax=axs[1])
    axs[1].set_title("Distribution of test set classes")
    plt.show()
    return X_train, X_test, y_train, y_test

def data_preprocessing_cnn(mnist: Bunch) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Extract input and target data from a sklearn Bunch data object
    :param mnist: A Bunch object containing the dataset
    :return: X_train, X_test, y_train, y_test as numpy arrays
    """
    X = mnist.data.values.astype('float32')  # Convert to float32
    y = mnist.target.values.astype('int64')  # Convert to int64
    X /= 255  # Normalize pixel values
    X = X.reshape(-1, 1, 28, 28)  # Reshape data to match the input shape
    test_split = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_split,
                                                        random_state=42)  # Split data into training and test sets

    return X_train, X_test, y_train, y_test


def plot_example(X: np.array, y: np.array):
    """Plot the first 5 images and their labels in a row."""
    print("Displaying examples. Close the figure windows to"
          "continue...")
    for i, (img, y) in enumerate(zip(X[:5].reshape(5, 28, 28), y[:5])):
        plt.subplot(151 + i)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(y)
    plt.suptitle("The first 5 examples of the training set")
    plt.show()

