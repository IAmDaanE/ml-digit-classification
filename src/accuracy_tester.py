import numpy as np
import nnlib as nn
from utils import load_mnist_csv

network = nn.Network(nn.Losses.softmax_cross_entropy)
network.add(nn.PreTrainedLayer("../models/v1/layer_0_weights.npy", "../models/v1/layer_0_biases.npy", nn.Activations.relu))
network.add(nn.PreTrainedLayer("../models/v1/layer_1_weights.npy", "../models/v1/layer_1_biases.npy", nn.Activations.relu))
network.add(nn.PreTrainedLayer("../models/v1/layer_2_weights.npy", "../models/v1/layer_2_biases.npy", nn.Activations.linear))

images, labels = load_mnist_csv("../data/mnist_test.csv")

predictions = network.forward(images)
guessed_numbers = np.argmax(predictions, axis=1)
actual_numbers = np.argmax(labels, axis=1)

accuracy = np.mean(guessed_numbers == actual_numbers) * 100
print(f"accuracy: {accuracy:.2f}%")