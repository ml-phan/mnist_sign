import argparse
import sys

from sklearn.model_selection import train_test_split

from data_loading import *
from train import *
from test import *


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
        X, y = data_preprocessing(mnist)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=42)
        model = train_model_fc(X_train, y_train)
        evaluate(model, X_test, y_test)

        torch.save(model, r"best_models/fc_nn.pt")

    if args.cnn:
        print("Do something with your CNN implementation here!")

    if len(sys.argv) <= 2:
        print(parser.print_help())


if __name__ == '__main__':
    main()
