from perceptron import Perceptron
from perceptron_layer import Layer
from perceptron_netwerk import Netwerk

def main():
    """
    Main function
    """
    perceptron = Perceptron([1, 1], 0)


    perceptrons_lst = [Perceptron([1, 1], -2), Perceptron([1, 1], -1), Perceptron([-1, -1], 1)]
    layer = Layer(perceptrons_lst)


    perceptrons_lst = [Perceptron([1, 0, 0], -1), Perceptron([0, 1, 1], -2)]
    layer_2 = Layer(perceptrons_lst)
    netwerk = Netwerk([layer, layer_2])


    print(perceptron)
    print(layer.activate_layer([1, 0]))
    print(layer)
    print(netwerk.evaluate([1, 1]))
    print(netwerk)


main()