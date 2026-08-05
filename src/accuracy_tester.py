import numpy as np
import nnlib as nn
from utils import load_mnist_csv

network = nn.Network(nn.Losses.softmax_cross_entropy, 784, 128, 2, 10)
network.add(nn.PreTrainedLayer("../models/v1/layer_0_weights.npy", "../models/v1/layer_0_biases.npy", nn.Activations.relu))
network.add(nn.PreTrainedLayer("../models/v1/layer_1_weights.npy", "../models/v1/layer_1_biases.npy", nn.Activations.relu))
network.add(nn.PreTrainedLayer("../models/v1/layer_2_weights.npy", "../models/v1/layer_2_biases.npy", nn.Activations.linear))

images, labels = load_mnist_csv("../data/mnist_test.csv")

total_score = 0

for i in range(images.shape[0]):
    prediction = network.forward(images[0]) #np.array([0.05, 0.05, ..., 0.85])
    if np.argmax(prediction) == np.argmax(labels[i]):
        total_score += 1

accuracy = round(total_score / images.shape[0] * 100, 1)

print(f"accuracy: {accuracy}%")