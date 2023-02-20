class Layer:

    def __init__(self, perceptrons):
        """
        Initializes the layer.
        :param perceptrons: The perceptrons of the layer.
        """
        self.perceptrons = perceptrons

    def activate_layer(self, inputs):
        """
        Activates the layer with the given inputs.
        :param inputs: The inputs to the layer.
        :return: The outputs of the layer.
        """
        outputs = []
        for perceptron in self.perceptrons:
            outputs.append(perceptron.activate(inputs))
        return outputs

    def __str__(self) -> str:
        """
        Returns a string representation of the layer.
        """
        return f"Layer(perceptrons={', '.join([str(x) for x in self.perceptrons])})"
