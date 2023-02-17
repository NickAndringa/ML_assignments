class Netwerk:
    #Initialize the netwerk with a list of layers.
    def __init__(self, layers = []):
        self.layers = layers

    #Evaluate the netwerk with the given inputs.
    def evaluate(self, inputs):
        for layer in self.layers:
            inputs = layer.activate_layer(inputs)
        return inputs

    #Return a string of the netwerk.
    def __str__(self) -> str:
        new_line = ',\n\t'
        return f'Netwerk(layers={new_line.join([str(x) for x in self.layers])})'
    