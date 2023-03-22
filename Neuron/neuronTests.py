def test_neuron_results(p, p_input, p_expected):
    """
    Tests the results of the neuron.
    :param p: The neuron to test.
    :param p_input: The input to the neuron.
    :param p_expected: The expected output of the neuron.
    :return: True if the output of the neuron is equal to the expected output, False otherwise.
    """
    p.sigmoid_activate(p_input)
    p_output = p.prev_output
    return True if (round(p_output) if p_output != 0.5 else 1) == p_expected else False


def test_neuron_network_results(p_n, p_n_input, p_n_expected):
    """
    Tests the results of the neuron network.
    :param p_n: The neuron network to test.
    :param p_n_input: The input to the neuron network.
    :param p_n_expected: The expected output of the neuron network.
    :return: True if the output of the neuron network is equal to the expected output, False otherwise.
    """
    p_n.feed_forward(p_n_input)
    p_n_output = p_n.prev_output
    return True if [(round(x) if x != 0.5 else 1) for x in p_n_output] == p_n_expected else False