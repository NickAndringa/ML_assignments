import unittest
from sklearn.datasets import load_iris
from perceptron import Perceptron
from perceptron_layer import Layer
from perceptron_netwerk import Netwerk
import pandas
import random
random.seed(1802273)

class TestPerceptron(unittest.TestCase):
    def setUp(self):
        """
        Setup for the perceptrons
        """
        self.perceptron_and = Perceptron([1, 1], -1.5)
        self.perceptron_or = Perceptron([1, 1], -0.5)
        self.perceptron_invert = Perceptron([-1], 0.5)
        self.perceptron_nor = Perceptron([-1, -1], 0.5)
        self.perceptron_nand = Perceptron([-1, -1], 1.5)
        self.perceptron_xor = Perceptron([3, -3], 0, eta=0.1)

        """
        Setup for the XOR network
        """
        perceptrons_lst = [self.perceptron_and, self.perceptron_nor]
        layer = Layer(perceptrons_lst)
        perceptrons_lst = [self.perceptron_nor]
        layer_2 = Layer(perceptrons_lst)
        netwerk = Netwerk([layer, layer_2])
        self.perceptron_netwerk_xor = netwerk

        """
        Setup for the half adder network
        """
        perceptrons_lst = [Perceptron([1, 1], -2), Perceptron([1, 1], -1), Perceptron([-1, -1], 1)]
        layer = Layer(perceptrons_lst)
        perceptrons_lst = [Perceptron([1, 0, 0], -1), Perceptron([0, 1, 1], -2)]
        layer_2 = Layer(perceptrons_lst)
        netwerk = Netwerk([layer, layer_2])
        self.perceptron_half_adder = netwerk


        """
        setup for learning_rule
        """
        self.perceptron_learning_rule = Perceptron([-1, 1], 0, eta=0.1)
        self.iris_test = Perceptron([random.random() for _ in range(4)], random.random(), eta=0.1)

        """
        Iris dataset setup
        """

        iris = load_iris()
        self.iris_data = pandas.DataFrame(data=iris.data, columns=iris.feature_names)
        self.iris_data['target'] = pandas.Series(iris.target)
        self.iris_data = self.iris_data[self.iris_data.target != 2]

        
    def test_and(self):
        """
        Test the AND gate perceptron
        """
        self.assertEqual(self.perceptron_and.activate([0, 0]), 0)
        self.assertEqual(self.perceptron_and.activate([0, 1]), 0)
        self.assertEqual(self.perceptron_and.activate([1, 0]), 0)
        self.assertEqual(self.perceptron_and.activate([1, 1]), 1) 

    def test_or(self):
        """
        Test the OR gate perceptron
        """
        self.assertEqual(self.perceptron_or.activate([0, 0]), 0)
        self.assertEqual(self.perceptron_or.activate([0, 1]), 1)
        self.assertEqual(self.perceptron_or.activate([1, 0]), 1)
        self.assertEqual(self.perceptron_or.activate([1, 1]), 1)

    def test_invert(self):
        """
        Test the invert gate perceptron
        """
        self.assertEqual(self.perceptron_invert.activate([0]), 1)
        self.assertEqual(self.perceptron_invert.activate([1]), 0)    
            
    def test_nor(self):
        """
        Test the NOR gate perceptron
        """
        self.assertEqual(self.perceptron_nor.activate([0, 0]), 1)
        self.assertEqual(self.perceptron_nor.activate([0, 1]), 0)
        self.assertEqual(self.perceptron_nor.activate([1, 0]), 0)
        self.assertEqual(self.perceptron_nor.activate([1, 1]), 0)

    def test_nand(self):
        """
        Test the NAND gate perceptron
        """
        self.assertEqual(self.perceptron_nand.activate([0, 0]), 1)
        self.assertEqual(self.perceptron_nand.activate([0, 1]), 1)
        self.assertEqual(self.perceptron_nand.activate([1, 0]), 1)
        self.assertEqual(self.perceptron_nand.activate([1, 1]), 0)

    def test_xor(self):
        """
        Test the XOR gate network
        """
        self.assertEqual(self.perceptron_netwerk_xor.evaluate([0, 0]), [0])
        self.assertEqual(self.perceptron_netwerk_xor.evaluate([0, 1]), [1])
        self.assertEqual(self.perceptron_netwerk_xor.evaluate([1, 0]), [1])
        self.assertEqual(self.perceptron_netwerk_xor.evaluate([1, 1]), [0])

    def test_half_adder(self):
        """
        Test the half adder network
        """
        self.assertEqual(self.perceptron_half_adder.evaluate([0, 0]), [0, 0])
        self.assertEqual(self.perceptron_half_adder.evaluate([0, 1]), [0, 1])
        self.assertEqual(self.perceptron_half_adder.evaluate([1, 0]), [0, 1])
        self.assertEqual(self.perceptron_half_adder.evaluate([1, 1]), [1, 0])
    
    def test_learning_and(self):
        """
        Test the learning of the AND gate
        """
        correct = False
        input_out =[
            [[0, 0], 0],
            [[1, 0], 0],
            [[0, 1], 0],
            [[1, 1], 1]
        ]
        while not correct:
            for io in input_out:
                self.perceptron_learning_rule.update(io[0], io[1])
                self.perceptron_learning_rule.loss(io[0], io[1])

            outputs = [self.perceptron_learning_rule.activate(io[0]) for io in input_out]

            results = [True if output == io[1] else False for output, io in zip(outputs, input_out)]


            if all(results):
                correct = True

        self.assertEqual(results, [True, True, True, True])

    def test_learning_xor(self):
        """
        Test the learning of the XOR gate
        """
        correct = False
        input_out =[
            [[0, 0], 0],
            [[1, 0], 1],
            [[0, 1], 1],
            [[1, 1], 1]
        ]
        while not correct:
            for io in input_out:
                self.perceptron_xor.update(io[0], io[1])
                self.perceptron_xor.loss(io[0], io[1])

            outputs = [self.perceptron_xor.activate(io[0]) for io in input_out]

            results = [True if output == io[1] else False for output, io in zip(outputs, input_out)]


            if all(results):
                correct = True

        self.assertEqual(results, [True, True, True, True])

    def test_learning_iris(self):
        """
        Test the learning of the iris dataset
        """
        correct = False

        while not correct:
            for index, row in self.iris_data.iterrows():
                self.iris_test.update([
                    row["sepal length (cm)"],
                    row["sepal width (cm)"],
                    row["petal length (cm)"],
                    row["petal width (cm)"],
                ], row["target"])

                self.iris_test.loss([
                    row["sepal length (cm)"],
                    row["sepal width (cm)"],
                    row["petal length (cm)"],
                    row["petal width (cm)"],
                ], row["target"])
            
            outputs = [self.iris_test.activate([
                    row["sepal length (cm)"],
                    row["sepal width (cm)"],
                    row["petal length (cm)"],
                    row["petal width (cm)"],
                ]) for index, row in self.iris_data.iterrows()]
            
            results = [True if output == row[1]["target"] else False for output, row in zip(outputs, self.iris_data.iterrows())]


            if all(results):
                correct = True
            else:
                correct = False
        
        self.assertEqual(correct, True)

