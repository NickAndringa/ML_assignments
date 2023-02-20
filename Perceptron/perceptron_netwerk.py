class Netwerk:

    def __init__(self, layers = []):
        """
        Initializes the netwerk.
        :param layers: The layers of the netwerk.
        """
        self.layers = layers

    def evaluate(self, inputs):
        """
        Evaluates the netwerk with the given inputs.
        :param inputs: The inputs to the netwerk.
        :return: The inputs for the next layer.
        """
        for layer in self.layers:
            inputs = layer.activate_layer(inputs)
        return inputs

    def __str__(self) -> str:
        """
        Returns a string representation of the netwerk.
        """
        new_line = ',\n\t'
        return f'Netwerk(layers={new_line.join([str(x) for x in self.layers])})'
    