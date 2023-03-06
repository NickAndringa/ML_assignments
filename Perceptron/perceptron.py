class Perceptron:

    def __init__(self, weights, bias, eta=0.1,):
        """
        Initializes the perceptron.
        :param weights: The weights of the perceptron.
        :param bias: The bias of the perceptron.
        """
        self.weights = weights
        self.bias = bias
        self.eta = eta
        self.error = None

           
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
    
    def update(self, inputs, d):
        """
        Updates the weights and bias of the perceptron.
        :param inputs: The inputs to the perceptron.
        :param d: The desired output of the perceptron.
        """

        y = self.activate(inputs)
        self.error = d - y

        # eta * error * input

        weight_deltas = [self.eta * self.error * input for input in inputs]
        self.weights = [sum(x) for x in zip(self.weights, weight_deltas)]

        # Resultaat opslaan in self.weighted_deltas
        # Delta's aan de weights toevoegen

        # eta * error
        # Reultaat opslaan in bias_delta
        # bias + bias_delta

        bias_delta = self.eta * self.error
        self.bias += bias_delta


        # weights = 0
        # for i in learning_rate[1]:
        # output = self.activate(inputs)

        return
    
    def loss(self, inputs, d):
        """
        Calculates the loss of the perceptron.
        :param inputs: The inputs to the perceptron.
        :param d: The desired output of the perceptron.
        """
        self.mse = sum([(d - (w * x)) ** 2 for w, x in zip(self.weights, inputs)]) / len(inputs)

    def __str__(self) -> str:
        """
        Returns a string representation of the perceptron.
        """
        return f"Perceptron(weights={self.weights}, bias={self.bias})"
    