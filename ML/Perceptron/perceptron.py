class Perceptron:
    
    #Initializes the perceptron
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
           
   #Returns 1 when x >= 0 otherwhise 0.
    def step(self, x):
        return int(x >= 0)
    
    #Activates the perceptron with the given inputs.
    def activate(self, inputs):
        for i in range(len(inputs)):
            inputs[i] *= self.weights[i]
        return self.step(sum(inputs) + self.bias)

    #Return a string of the perceptron.
    def __str__(self) -> str:
        return f"Perceptron(weights={self.weights}, bias={self.bias})"
    