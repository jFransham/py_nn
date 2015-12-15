__author__ = 'Jack'
import numpy as np
import random

def USE_BIAS():
    return False


class NeuralNetwork:
    def __init__(self, activation, dif_activation, inputs, first, *rest):
        self.activation = activation
        self.dif_activation = dif_activation
        self.inputs = inputs
        self.num_nodes = [first] + list(rest)

        self.outputs = self.num_nodes[-1]
        # + 1 to input ranges so that there can be a hidden "bias" node
        self.weights = \
            [
                np.mat([[random.random() for _ in range(inputs + (1 if USE_BIAS() else 0))]
                        for _ in range(self.num_nodes[0])])
            ] + \
            [
                np.mat([[random.random() for _ in range(self.num_nodes[p - 1] + (1 if USE_BIAS() else 0))]
                        for _ in range(self.num_nodes[p])])
                for p in range(1, len(self.num_nodes))
            ]

    def get_output(self, i_raw):
        return self.get_activations(i_raw)[-1]

    def backprop(self, i_raw, error_func, dif_error_func, rate):
        col, raw_col = self.get_activations_internal(i_raw)
        activations, raw_activations = ([NeuralNetwork.col_to_arr(c) for c in a] for a in (col, raw_col))

        inputs = [NeuralNetwork.get_flat_bias(a) for a in [i_raw] + raw_activations[:-1]]

        last = len(self.num_nodes) - 1

        errors      = ([0] * (len(self.num_nodes) - 1)) + \
            [dif_error_func(activations[last]) * np.array(self.dif_activation(raw_activations[last]))]
        derivatives = ([0] * (len(self.num_nodes) - 1)) + \
            [np.mat(errors[last]).transpose() * np.mat(inputs[last])]

        if len(self.num_nodes) > 1:
            for counter in reversed(range(len(self.num_nodes) - 1)):
                errors[counter] = self.dif_activation(raw_activations[counter]) * \
                    NeuralNetwork.col_to_arr(
                        self.weights[counter + 1].transpose() *
                        NeuralNetwork.arr_to_col(errors[counter + 1])
                    )
                derivatives[counter] = np.mat(errors[counter]).transpose() * np.mat(inputs[counter])

        return [rate * d for d in derivatives]

    def get_activations(self, i_raw):
        a, _ = self.get_activations_internal(i_raw)
        return [list(NeuralNetwork.col_to_arr(m)) for m in a]

    def get_activations_internal(self, i_raw):
        if len(i_raw) != self.inputs:
            raise Exception("Input wrong size")

        a, r = self.get_activation_layer(0, NeuralNetwork.arr_to_col(i_raw))
        activations = [a]
        raw_activations = [r]
        for counter in range(1, len(self.num_nodes)):
            a, r = self.get_activation_layer(counter, activations[counter - 1])
            activations.append(a)
            raw_activations.append(r)

        return activations, raw_activations

    def get_activation_layer(self, layer, col):
        raw = self.weights[layer] * NeuralNetwork.get_with_bias(col)
        return self.activation(raw), raw

    @staticmethod
    def get_with_bias(i):
        return NeuralNetwork.arr_to_col(NeuralNetwork.get_flat_bias(i))

    @staticmethod
    def get_flat_bias(i):
        return list(NeuralNetwork.col_to_arr(i)) + ([1] if USE_BIAS() else [])

    @staticmethod
    def get_flat_differentiation_bias(i):
        return list(NeuralNetwork.col_to_arr(i)) + ([0] if USE_BIAS() else [])

    @staticmethod
    def col_to_arr(v):
        return np.array(v).flatten()

    @staticmethod
    def arr_to_col(l):
        return np.transpose(np.mat(l))

num = 10

n = NeuralNetwork(# lambda x: np.tanh(x),
                  lambda x: 1 / (1 + np.exp(-x)),
                  # lambda x: 1 - np.tanh(x) * np.tanh(x),
                  lambda x: np.exp(x) / pow(np.exp(x) + 1, 2),
                  num,
                  num,
                  num,
                  num)

inp = 0
mx = 1.0
itr = 20000
for j in range(itr):
    rng = range(num)
    random.shuffle(rng)
    inp = (1.0/num) * np.array(rng)
    # for k in range(50):
    t = np.array(list(reversed(inp))) # np.array([0] * n.num_nodes[-1])

    difs = n.backprop(
        inp,
        lambda k: [sum((t - k) * (t - k)) / 2] * len(inp),
        lambda k: t - k,
        0.2)

    n.weights = [n.weights[i] + difs[i] for i in range(len(n.num_nodes))]

print inp
acts = n.get_activations(inp)
for a in acts:
    print a

#print n.get_output(inp)
