from datasets import load_dataset
import numpy as np

def load_mnist():

    dataset = load_dataset("ylecun/mnist")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    X_train = np.array(train_dataset["image"]) / 255.0
    y_train = np.array(train_dataset["label"])

    X_test = np.array(test_dataset["image"]) / 255.0
    y_test = np.array(test_dataset["label"])

    return X_train, y_train, X_test, y_test
