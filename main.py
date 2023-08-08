import argparse
import sys

from utils.data_loading import *
from model.train import *
from model.test import *


def main():
    parser = argparse.ArgumentParser(
        "Training model on MNIST sign language dataset")
    parser.add_argument("-fc", action='store_true',
                        help="using fully connected neural network "
                             "architecture")
    parser.add_argument("-cnn", action='store_true',
                        help="using convolutional neural network architecture")
    args = parser.parse_args()

    if args.fc:
        mnist = get_dataset_openml()
        X_train, X_test, y_train, y_test = data_preprocessing(mnist)
        plot_example(X_train, y_train)
        model = train_model_fc(X_train, y_train)
        evaluate(model, X_test, y_test)
        torch.save(model, r"best_models/fc_nn.pt")

    if args.cnn:
        mnist = get_dataset_openml()
        X_train, X_test, y_train, y_test = data_preprocessing_cnn(mnist)
        model = train_model_CNN(X_train, y_train)
        evaluate_mode(model, X_test, y_test)
        torch.save(model, r"best_models/cnn.pt")

    if len(sys.argv) <= 2:
        print(parser.print_help())


if __name__ == '__main__':
    main()
