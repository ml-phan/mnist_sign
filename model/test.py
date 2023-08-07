from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

