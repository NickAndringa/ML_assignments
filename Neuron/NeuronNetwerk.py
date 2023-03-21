class NeuronNetwerk:

    def __init__(self, layers = []):
        """
        Initializes the netwerk.
        :param layers: The layers of the netwerk.
        """
        self.layers = layers
        self.prev_output = None

    def feed_forward(self, inputs):
        """
        Evaluates the netwerk with the given inputs.
        :param inputs: The inputs to the netwerk.
        """
        for layer in self.layers:
            inputs = layer.activate_layer(inputs)

        self.prev_output = inputs

    def get_feed_forward_results(self, inputs):
        """
        Returns the results of the feed forward.
        :param inputs: The inputs to the netwerk.
        :return: The results of the feed forward.
        """
        self.feed_forward(inputs)
        return self.prev_output

    def backward_prob(self, targets, eta = 0.1):
        """
        Backpropagates the netwerk.
        :param targets: The targets of the netwerk.
        :param eta: The learning rate.
        """
        self.layers[-1].calc_output_error(targets)
        for i in reversed(range(0, len(self.layers))):
            self.layers[i].train(eta)
            error = self.layers[i].get_hidden_errors()

            if i > 0:
                self.layers[i - 1].assign_errors(error)

    def update(self):
        """
        Updates the netwerk.
        """
        for layer in self.layers:
            layer.update()

    def train(self, inputs, targets, eta = 0.1, max_epochs = 10000):
        """
        Trains the netwerk.
        :param inputs: The inputs to the netwerk.
        :param targets: The targets of the netwerk.
        :param eta: The learning rate.
        :param max_epochs: The maximum amount of epochs.
        """
        correct = False
        epoch = 0

        while not correct:
            epoch += 1
            for i, t in zip(inputs, targets):
                self.feed_forward(i)

                self.backward_prob(t, eta)

                self.update()

            if epoch >= max_epochs:
                break

    def __str__(self) -> str:
        """
        Returns a string representation of the netwerk.
        """
        new_line = ',\n\t'
        return f'Netwerk(layers={new_line.join([str(x) for x in self.layers])})'
    