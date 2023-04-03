import math
import random

class Neuron:
    def __init__(self, weights, bias, eta=0.1):
        """
        Initializes the neuron
        :param weights: the weight of thd neuron
        :param bias: the bias of the neuron
        """
        self.weights = weights
        self.bias = bias
        self.eta = eta
        self.prev_outputs = 0
        self.prev_inputs = 0
        self.delta = 0
        self.output_weights = 0


    def activate(self, inputs):
        """
        Activates the neuron with the given inputs.
        :param inputs: The inputs to the neuron.
        :return: The output of the neuron.
        """
        self.prev_inputs = inputs
        return self.bias + sum([w * x for w, x in zip(self.weights, inputs)])

    def gradients(self):
        """
        Calculates the gradients of the neuron.
        :return: The gradients of the neuron.
        """
        return [prev_i * self.error for prev_i in self.prev_inputs]

    def sigmoid_activation(self, inputs):
        """
        Activates the neuron with the sigmoid function.
        :params inputs: the inputs to the neuron
        """
        self.prev_outputs = 1/(1 + math.exp(-(self.activate(inputs))))
    
    def get_sigmoid_results(self, inputs):
        """
        Returns the results from the sigmoid function.
        :params inputs: the inputs to the neuron
        :return: sigmoid function results
        """
        self.sigmoid_activation(inputs)
        return self.prev_outputs
    
    def calc_output_error(self, target): 
        """
        Calculates the error of the output neuron.
        :params target: 
        """
        self.error = self.prev_outputs * (1 - self.prev_outputs) * - (target - self.prev_outputs)
        # return self.error


        # activeer
        # sla temp input en output op in feed forward

        # bereken error met delta rule
        # sla op in Neuron
        
        # = (afeleide van sigmoid) * - (target - output) =
        # = output * (1 - output) * - (target - outputs) =

        # return error

    def get_hidden_errors(self):
        """
        Returns the errors of the hidden neurons.
        :return: the errors of the hidden neurons.
        """
        # hidden_error = self.prev_outputs - (1 - self.prev_outputs) * sum([w * x for w, x in zip(deltas, output_weights)])

        hidden_error = [w * self.error for w in self.weights]
        return hidden_error

    def train(self, eta):
        """
        Trains the neuron.
        :param eta: the learning rate
        """
        self.sigmoid_activation(self.prev_inputs)

        self.delta_weights = [eta * gradient for gradient in self.gradients()]

        self.delta_bias = eta * (1 * self.error)

    def update(self):
        """
        Updates the weights and bias of the neuron.
        """
        self.weights = [w - dw for w, dw in zip(self.weights, self.delta_weights)]
    
        self.bias -= self.delta_bias

    def calc_hidden_error(self, errors):
        """
        Calculates the error of the hidden neuron.
        :params errors: the errors of the hidden neurons
        """
        self.error = self.prev_outputs * (1 - self.prev_outputs) * sum([e for e in errors])
    
    def __str__(self):
        """
        Returns a string representation of the neuron.
        """
        return f"Neuron(weights={self.weights}, bias={self.bias})"
