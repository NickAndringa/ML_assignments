import math

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        return self.bias + sum([w * x for w, x in zip(self.weights, inputs)])

    def sigmoid_activation(self, inputs):
        return 1/(1 + math.exp(-(self.activate(inputs))))
    
    def __str__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias})"