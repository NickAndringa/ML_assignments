class Layer:
    #Initialize the layer with a list of perceptrons
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons

    #Activate the layer with the given inputs
    def activate_layer(self, inputs):
        outputs = []
        for perceptron in self.perceptrons:
            outputs.append(perceptron.activate(inputs))
        return outputs

    #Return a string of the layer
    def __str__(self) -> str:
        return f"Layer(perceptrons={', '.join([str(x) for x in self.perceptrons])})"
