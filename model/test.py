from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch



def evaluate(model, X_test: np.array, y_test: np.array):
    y_pred = model.predict(X_test)
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred)}")
    print("Displaying classification report. Close the figure windows to"
          "continue...")
    cl_report = classification_report(y_test, y_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(cl_report).iloc[:-1, :].T, annot=True,
                cmap="Blues")
    plt.suptitle("Classification report on the test set")
    plt.tight_layout()
    plt.show()

    print("Displaying confusion matrix. Close the figure windows to"
          "exit.")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    disp.plot(ax=ax, cmap="Blues")
    plt.suptitle("Confusion matrix on the test set")
    plt.tight_layout()
    plt.show()

def evaluate_mode(model, train_data, test_data):
    def calculate_accuracy(data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total * 100
        return accuracy
    
    train_accuracy = calculate_accuracy(train_data)
    test_accuracy = calculate_accuracy(test_data)
    
    print("Train Data Accuracy: {:.2f}%".format(train_accuracy))
    print("Test Data Accuracy: {:.2f}%".format(test_accuracy))
    