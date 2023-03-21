def test_neuron_results(p, p_input, p_expected):
    p.sigmoid_activate(p_input)
    p_output = p.prev_output
    return True if (round(p_output) if p_output != 0.5 else 1) == p_expected else False


def test_neuron_network_results(p_n, p_n_input, p_n_expected):
    p_n.feed_forward(p_n_input)
    p_n_output = p_n.prev_output
    return True if [(round(x) if x != 0.5 else 1) for x in p_n_output] == p_n_expected else False