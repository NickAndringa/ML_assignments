class NeuronLayer:

    def __init__(self, neurons):
        """
        Initializes the layer.
        :param neurons: The neurons of the layer.
        """
        self.neurons = neurons

    def activate_layer(self, inputs):
        """
        Activates the layer with the given inputs.
        :param inputs: The inputs to the layer.
        :return: The outputs of the layer.
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.get_sigmoid_results(inputs))
        return outputs
    
    def calc_output_error(self, targets):
        """
        Calculates the error of the output layer.
        :param targets: The targets of the output layer.
        """
        for neuron, target in zip(self.neurons, targets):
            neuron.calc_output_error(target)
    
    def train(self, eta = 0.1):
        """
        Trains the layer.
        :param eta: The learning rate.
        """
        for i in range(len(self.neurons)):
            self.neurons[i].train(eta)
    
    def get_hidden_errors(self):
        """
        Returns the errors of the hidden layer.
        :return: The errors of the hidden layer.
        """
        return [neuron.get_hidden_errors() for neuron in self.neurons]
    
    def assign_errors(self, errors):
        """
        Assigns the errors to the neurons.
        :param errors: The errors to assign.
        """
        sorted_errors = [[] for _ in self.neurons]

        for i, n in enumerate(self.neurons):
            for j in range(len(errors)):
                sorted_errors[i].append(errors[j][i])

        for i, n in enumerate(self.neurons):
            n.calc_hidden_error(sorted_errors[i])

    def update(self):
        """
        Updates the layer.
        """
        for neuron in self.neurons:
            neuron.update()

    def __str__(self) -> str:
        """
        Returns a string representation of the layer.
        """
        return f"Layer(neurons={', '.join([str(x) for x in self.neurons])})"
