import numpy as np
import nnlib as nn
from utils import load_mnist_csv, shuffle_dataset

network = nn.Network(nn.Losses.softmax_cross_entropy, 784, 128, 2, 10, 1280, 650)
network.add(nn.Layer(784, 128, nn.Activations.relu))
network.add(nn.Layer(128, 128, nn.Activations.relu))
network.add(nn.Layer(128, 10, nn.Activations.linear))

images, labels = load_mnist_csv("../data/mnist_train.csv")

batch_size = 64

def run_training(epochs):
    global images, labels
    current_lr = 0.1
    try:
        for epoch in range(epochs):
            network.epoch = epoch
            current_lr = nn.LrDecays.exponential_decay(current_lr, 0.955)
            network.current_lr = current_lr
            images, labels = shuffle_dataset(images, labels)
            for i in range(0, images.shape[0], batch_size):
                images_batch = images[i : i + batch_size]
                labels_batch = labels[i : i + batch_size]
                prediction = network.forward(images_batch)
                loss = network.loss_function(prediction, labels_batch)
                network.loss = loss
                network.backward(prediction, labels_batch)
                network.update(current_lr)
                if network.screen:
                    network.check_pygame_events()
            network.visualize()
    except KeyboardInterrupt:
        for i, layer in enumerate(network.layers):
            np.save(f"../models/v2/layer_{i}_weights.npy", layer.weights)
            np.save(f"../models/v2/layer_{i}_biases.npy", layer.biases)
        return

if __name__ == "__main__":
    run_training(30000)