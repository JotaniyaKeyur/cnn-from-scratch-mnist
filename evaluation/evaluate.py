import numpy as np

def evaluate(model, X_test, y_test):

    correct = 0

    for i in range(len(X_test)):

        X = X_test[i]

        y_pred = model.forward(X)

        pred_class = np.argmax(y_pred)

        if pred_class == y_test[i]:
            correct += 1

    test_accuracy = correct / len(X_test)

    print(f"Test Accuracy: {test_accuracy}")
