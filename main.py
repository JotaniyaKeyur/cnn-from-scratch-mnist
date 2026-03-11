from data.dataset import load_mnist
from model.cnn import MultiClassClassification
from training.trainer import train
from evaluation.evaluate import evaluate

def main():

    X_train, y_train, X_test, y_test = load_mnist()

    model = MultiClassClassification(lr=0.01)

    train(model, X_train, y_train, epochs=5)

    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()
