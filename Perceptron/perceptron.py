class Perceptron:

    def __init__(self, weights, bias):
        """
        Initializes the perceptron.
        :param weights: The weights of the perceptron.
        :param bias: The bias of the perceptron.
        """
        self.weights = weights
        self.bias = bias
           
    def step(self, x):
        """
        Returns 1 when x >= 0 otherwise 0.
        :param x: The value to check.
        :return: 1 when x >= 0 otherwise 0.
        """
        return int(x >= 0)
    
    def activate(self, inputs):
        """
        Activates the perceptron with the given inputs.
        :param inputs: The inputs to the perceptron.
        :return: The output of the perceptron.
        """
        temp_inputs = inputs.copy()
        for i in range(len(temp_inputs)):
            temp_inputs[i] *= self.weights[i]
        return self.step(sum(temp_inputs) + self.bias)

    def __str__(self) -> str:
        """
        Returns a string representation of the perceptron.
        """
        return f"Perceptron(weights={self.weights}, bias={self.bias})"
    