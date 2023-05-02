# imported libraries
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import datasets
from math import e
import torch

# constants
TRAIN_NUM, TEST_NUM, LABEL_NUM = 60000, 10000, 10
IMG_SIZE, IMG_VEC_LEN = 28, 784
SKIP = 5


def load_MNIST(batch_size, shuffle):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="data",
                                   train=True,
                                   transform=transform,
                                   download=True)

    test_dataset = datasets.MNIST(root="data",
                                  train=False,
                                  transform=transform,
                                  download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def plot_graph(train, test, epochs):
    plt.plot(epochs, train, c='b', label="train")
    plt.plot(epochs, test, c='r', label="test")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train Accuracy VS Test Accuracy as a function of Epochs Number")

    plt.legend()
    plt.show()


# utilities functions
sigmoid = lambda x: 1 / (1 + (e ** -x))
sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
soft_max = lambda vec: (e ** vec) / torch.sum(e ** vec)


def soft_max_function(Z):
    P = torch.zeros((Z.shape))

    for i, row in enumerate(Z):
        P[i] = soft_max(row)

    return P


def create_one_hot_vectors_matrix(y, labels_num):
    Y = torch.zeros(y.size(0), labels_num)

    for yi, row_Y in zip(y, Y):
        row_Y[int(yi.item())] = 1

    return Y


class NeuralNetwork:

    def __init__(self, input_size=IMG_VEC_LEN, hidden_size=100, labels_num=10):
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.labels_num = labels_num

        # weights
        self.W1 = torch.randn(input_size, hidden_size)
        self.W2 = torch.randn(hidden_size, labels_num)

        # biases
        self.b1 = torch.zeros(hidden_size)
        self.b2 = torch.zeros(labels_num)

    def load(self, parameters_dic):
        # load weights
        self.W1 = parameters_dic["W1"]
        self.W2 = parameters_dic["W2"]

        # load biases
        self.b1 = parameters_dic["b1"]
        self.b2 = parameters_dic["b2"]

    def forward(self, X):
        # hidden layer
        self.Z1 = torch.mm(X, self.W1) + self.b1
        self.H1 = sigmoid(self.Z1)

        # final layer
        self.Z2 = torch.mm(self.H1, self.W2) + self.b2

        # probability matrix - (i,j) entry represents the probability of th ith records to be under the label j
        P = soft_max_function(self.Z2)

        return P

    def backward(self, X, y, P, lr=0.1):
        batch_size = y.size(0)
        Y = create_one_hot_vectors_matrix(y, labels_num=self.labels_num)

        # partial derivatives of the loss function
        # we used the cross entropy loss function (multiclass)
        dl_dZ2 = (1 / batch_size) * (P - Y)
        dl_dH1 = torch.matmul(dl_dZ2, torch.t(self.W2))
        dl_dZ1 = dl_dH1 * sigmoid_prime(self.Z1)

        # gradient descent with learning rate (step size) "lr"
        self.W1 -= lr * torch.matmul(torch.t(X), dl_dZ1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dZ1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.H1), dl_dZ2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dZ2), torch.ones(batch_size))

    def train_on_batch(self, X, y):
        P = self.forward(X)
        self.backward(X, y, P)

    def predict(self, X):
        P = self.forward(X)
        predicated_labels = torch.argmax(P, dim=1)

        return predicated_labels


def find_accuracy(model, data_loader, n):
    summ = 0

    for batch, labels in data_loader:
        batch = batch.view(-1, IMG_SIZE * IMG_SIZE)
        predicated_labels = model.predict(batch)

        for pred_label, label in zip(predicated_labels, labels):

            if pred_label == label:
                summ += 1

    acc = summ / n
    return acc


def train_one_epoch(model, train_loader):
    for j, (batch, labels) in enumerate(train_loader):
        batch = batch.view(-1, IMG_SIZE * IMG_SIZE)
        model.train_on_batch(batch, labels)


def train(model, train_loader, test_loader, epochs_num):
    train_acc, test_acc = [], []

    for i in range(epochs_num):
        print(f"Epoch: {i} \n")

        train_one_epoch(model, train_loader)

        # calculating the accuracy of the model on the train and test datasets at each "SKIP"-th epoch
        if i % SKIP == 0:
            train_acc.append(find_accuracy(model, train_loader, n=TRAIN_NUM))
            test_acc.append((find_accuracy(model, test_loader, n=TEST_NUM)))

    print("Training has ended successfully! \n")

    return train_acc, test_acc


def q1():
    train_loader, test_loader = load_MNIST(batch_size=128, shuffle=True)
    neural_network = NeuralNetwork(input_size=IMG_VEC_LEN, hidden_size=100, labels_num=LABEL_NUM)
    epochs_num = 51

    train_acc, test_acc = train(neural_network, train_loader, test_loader, epochs_num)

    print(f"The accuracy of the trained model on the test dataset: {round(test_acc[len(test_acc) - 1], 3)}")

    plot_graph(train_acc, test_acc, epochs=[i for i in range(0, epochs_num, SKIP)])

    # saving the trained model
    trained_model = {
        'W1': neural_network.W1,
        'W2': neural_network.W2,
        'b1': neural_network.b1,
        'b2': neural_network.b2
    }

    torch.save(trained_model, 'trained_model.pkl')


def main():
    q1()


if __name__ == "__main__":
    main()
