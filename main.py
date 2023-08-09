import argparse
import sys

from utils.data_loading import *
from model.train import *
from model.test import *
from model.architecture import *



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
        train_path = r"data/sign_mnist_train.csv"
        test_path = r"data/sign_mnist_test.csv"
        train_data = SMDataset(train_path)
        test_data = SMDataset(test_path)
        train_dl = DataLoader(train_data, batch_size=128, shuffle=True)
        test_dl = DataLoader(test_data, batch_size=32)
        # Create the CNN model
        model_cnn = SMCNN()
        # Train the CNN model
        epoch_data, loss_data = train_model_CNN(train_dl, model_cnn, n_epochs=1)
        evaluate_mode(train_data,test_data,model_cnn)
        torch.save(model_cnn, r"best_models/cnn.pt")

    if len(sys.argv) <= 2:
        print(parser.print_help())


if __name__ == '__main__':
    main()
