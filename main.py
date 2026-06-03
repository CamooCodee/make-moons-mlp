import sklearn
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

def main():
    N = 200
    data = sklearn.datasets.make_moons(n_samples=N, noise=0.2, random_state=67)
    (x, y) = data
    npdata = np.column_stack(data)

    features = npdata[:, :2]

    mean = features.mean(axis=0)
    std = features.std(axis=0)

    features -= mean
    features /= std

    c: list[str] = []

    for i in range(0, N):
        c.append("red" if y[i] == 0 else "green")

    plt.scatter(x=npdata[:, 0], y=npdata[:, 1], c=c, s=5)

    print("-- Data")
    print(npdata.shape)
    print("--\n\n")

    neural_net(npdata)

    # plt.show()

def neural_net(data: np.ndarray):
    hidden_layer_neurons = 6
    input_count = data.shape[1] - 1 # -1 since the last column is the ground truth

    rng = default_rng(67)

    weight_layers = []
    bias_layers = []

    weight_layers.append(rng.uniform(low=-1, high=1, size=(hidden_layer_neurons, input_count)))
    bias_layers.append(rng.uniform(low=-1, high=1, size=hidden_layer_neurons))

    weight_layers.append(rng.uniform(low=-1, high=1, size=(hidden_layer_neurons, hidden_layer_neurons)))
    bias_layers.append(rng.uniform(low=-1, high=1, size=hidden_layer_neurons))

    weight_layers.append(rng.uniform(low=-1, high=1, size=(1, hidden_layer_neurons)))
    bias_layers.append(np.array([0.67]))

    for i in range(2000):
        # Forward Pass

        x = data[:, :2]
        out = x.T
        layers = len(weight_layers)
        pre_activ_layer_outputs = []

        for j in range(layers):
            input = out
            weights = weight_layers[j]
            biases = bias_layers[j]
            out = weights @ input + biases[:, np.newaxis]

            pre_activ_layer_outputs.append(out.copy())

            if j == layers - 1:
                out = sigmoid(out)
            else:
                out = np.tanh(out)

        out = np.squeeze(out)


        L = loss(data[:, -1], out)

        if i % 50 == 0:
            print(f"{i}. Loss: {L}")

        # Backward Pass
        
        gt = data[:, -1]
        weight_grads = []
        bias_grads = []

        # z = pre activiation
        z_grad = (sigmoid(pre_activ_layer_outputs[-1]) - gt) / data.shape[0]
        bias_grads.insert(0, z_grad.sum())
        weight_grads.insert(0, z_grad @ np.tanh(pre_activ_layer_outputs[-2]).T)
        x_grad = z_grad.T @ weight_layers[-1]

        z_grad = (1 - np.square(np.tanh(pre_activ_layer_outputs[-2].T))) * x_grad
        bias_grads.insert(0, z_grad.sum(axis=0))
        weight_grads.insert(0, z_grad.T @ np.tanh(pre_activ_layer_outputs[-3]).T)
        x_grad = z_grad @ weight_layers[-2]

        z_grad = (1 - np.square(np.tanh(pre_activ_layer_outputs[-3].T))) * x_grad
        bias_grads.insert(0, z_grad.sum(axis=0))
        weight_grads.insert(0, z_grad.T @ x)

        # Apply

        lr = 3

        for i in range(len(bias_grads)):
            bias_layers[i] -= bias_grads[i] * lr

        for i in range(len(weight_grads)):
            weight_layers[i] -= weight_grads[i] * lr

    print(out)

# Binary Cross Entropy
def loss(gt: np.ndarray, out: np.ndarray) -> float:
    return -(gt * np.log(out) + (1 - gt) * np.log(1 - out)).mean()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    main()
