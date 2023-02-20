import unittest
from perceptron import Perceptron
from perceptron_layer import Layer
from perceptron_netwerk import Netwerk

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

        """
        Setup for the XOR network
        """
        perceptrons_lst = [self.perceptron_and, self.perceptron_nor]
        layer = Layer(perceptrons_lst)
        perceptrons_lst = [self.perceptron_nor]
        layer_2 = Layer(perceptrons_lst)
        netwerk = Netwerk([layer, layer_2])
        self.perceptron_xor = netwerk

        """
        Setup for the half adder network
        """
        perceptrons_lst = [Perceptron([1, 1], -2), Perceptron([1, 1], -1), Perceptron([-1, -1], 1)]
        layer = Layer(perceptrons_lst)
        perceptrons_lst = [Perceptron([1, 0, 0], -1), Perceptron([0, 1, 1], -2)]
        layer_2 = Layer(perceptrons_lst)
        netwerk = Netwerk([layer, layer_2])
        self.perceptron_half_adder = netwerk
        
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
        self.assertEqual(self.perceptron_xor.evaluate([0, 0]), [0])
        self.assertEqual(self.perceptron_xor.evaluate([0, 1]), [1])
        self.assertEqual(self.perceptron_xor.evaluate([1, 0]), [1])
        self.assertEqual(self.perceptron_xor.evaluate([1, 1]), [0])

    def test_half_adder(self):
        """
        Test the half adder network
        """
        self.assertEqual(self.perceptron_half_adder.evaluate([0, 0]), [0, 0])
        self.assertEqual(self.perceptron_half_adder.evaluate([0, 1]), [0, 1])
        self.assertEqual(self.perceptron_half_adder.evaluate([1, 0]), [0, 1])
        self.assertEqual(self.perceptron_half_adder.evaluate([1, 1]), [1, 0])
        
