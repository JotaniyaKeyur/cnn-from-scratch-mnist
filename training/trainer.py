import numpy as np

def train(model, X_train, y_train, epochs=5):

    for epoch in range(epochs):

        total_loss = 0
        correct = 0

        for i in range(len(X_train)):

            X = X_train[i]
            label = y_train[i]

            y_pred = model.forward(X)

            y_true = np.zeros((10, 1))
            y_true[label] = 1

            loss = -np.sum(y_true * np.log(y_pred + 1e-12))
            total_loss += loss

            pred_class = np.argmax(y_pred)

            if pred_class == label:
                correct += 1

            model.backward(y_true)
            model.update()

        avg_loss = total_loss / len(X_train)
        accuracy = correct / len(X_train)

        print(f"Epoch: {epoch+1}, Loss: {avg_loss}, Accuracy: {accuracy}")
