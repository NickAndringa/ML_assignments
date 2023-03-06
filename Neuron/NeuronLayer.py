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
            outputs.append(neuron.sigmoid_activation(inputs))
        return outputs

    def __str__(self) -> str:
        """
        Returns a string representation of the layer.
        """
        return f"Layer(neurons={', '.join([str(x) for x in self.neurons])})"
