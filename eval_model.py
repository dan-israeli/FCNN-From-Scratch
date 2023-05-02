# imported functions and classes
from train_model import load_MNIST, find_accuracy, NeuralNetwork
import torch

# constants
TEST_NUM, IMG_SIZE = 10000, 28


def evaluate_hw1_q1():
    # load test dataset
    _, test_loader = load_MNIST(batch_size=128, shuffle=False)

    # load trained model
    neural_network = NeuralNetwork()
    trained_model = torch.load("trained_model.pkl")
    neural_network.load(trained_model)

    # find accuracy on the test dataset
    acc = find_accuracy(neural_network, test_loader, n=TEST_NUM)
    print(f"The accuracy of the trained model on the test dataset: {round(acc, 3)}")


def main():
    evaluate_hw1_q1()


if __name__ == "__main__":
    main()
