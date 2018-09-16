import numpy as np
import matplotlib.pyplot as plt
from first_neural.neural_network import NeuralNetwork


def get_train_data_list():
    data_file = open('../data/mnist_dataset/mnist_train_100.csv', 'r')
    # data_file = open('../data/mnist_dataset/mnist_train.csv', 'r')
    data_list = data_file.readlines()

    data_file.close()

    return data_list


def get_test_data_list():
    data_file = open('../data/mnist_dataset/mnist_test_10.csv', 'r')
    # data_file = open('../data/mnist_dataset/mnist_test.csv', 'r')
    data_list = data_file.readlines()

    data_file.close()

    return data_list


def show_mnist_image(data_list, index):
    all_values = data_list[index].split(',')

    image_array = np.asfarray(all_values[1:]).reshape((28, 28))

    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()


# show_mnist_image(data_list, 2)

if __name__ == '__main__':
    input_nodes = 784
    output_nodes = 10
    epochs = 5
    data_list = get_train_data_list()
    test_data_list = get_test_data_list()
    n = NeuralNetwork(input_nodes, 100, output_nodes, 0.3)
    for e in range(epochs):
        for record in data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    # show_mnist_image(test_data_list, 0)
    # all_values = test_data_list[0].split(',')
    # result = n.query((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    # print(result)

    scorecard = []

    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)

        label = np.argmax(outputs)
        print(label, "network's answer")

        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = np.asarray(scorecard)

    print("performance = ", scorecard_array.sum() / scorecard_array.size)
