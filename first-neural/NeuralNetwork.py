import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        # self.weight_input_hidden = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        # self.weight_hidden_output = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        self.weight_ih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weight_ho = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.weight_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.weight_ho.T, output_errors)

        self.weight_ho += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                      np.transpose(hidden_outputs))

        self.weight_ih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                      np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.weight_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weight_ho, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.3

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print('test')
    print(n.query([1.0, 0.5, -1.5]))
