import numpy as np
import pygame
from torchvision.datasets import MNIST

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 900

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("network visualization")
clock = pygame.time.Clock()

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def linear(x):
    return x

def linear_grad(x):
    return np.ones_like(x)

def cross_entropy(prediction, true_value):
    eps = 1e-9
    return -np.sum(true_value * np.log(prediction + eps)) / len(prediction)

def cross_entropy_grad(prediction, true_value):
    return (prediction - true_value) / len(prediction)

def mse(prediction, true_value):
    return np.mean((prediction - true_value) ** 2)

def mse_grad(prediction, true_value):
    return 2 * (prediction - true_value) / prediction.size

def softmax(z):
    ez = np.exp(z - np.max(z, axis=1, keepdims=True))
    return ez / np.sum(ez, axis=1, keepdims=True)

activation_map = {
    relu: relu_grad,
    linear: linear_grad,
    mse: mse_grad,
    cross_entropy: cross_entropy_grad
}

class Layer:
    def __init__(self, n_in, n_out, activation):
        self.weights = np.random.randn(n_in, n_out) * 0.1
        #self.weights = np.ones((n_in, n_out)) * 0.1
        self.biases = np.zeros((1, n_out))
        self.activation = activation

    def forward(self, inputs):
        self.cached_inputs = inputs
        self.pre_activation = inputs @ self.weights + self.biases
        return

    def backward(self, incoming_gradient):
        activation_gradient = activation_map[self.activation](self.pre_activation)
        gradient_after_activation = incoming_gradient * activation_gradient
        self.weight_gradient = self.cached_inputs.T @ gradient_after_activation
        self.bias_gradient = np.sum(gradient_after_activation, axis=0, keepdims=True)
        return gradient_after_activation @ self.weights.T

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weight_gradient
        self.biases -= learning_rate * self.bias_gradient

class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, prediction, true_value):
        gradient = cross_entropy(prediction, true_value)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

network = Network()
network.add(Layer(1, 16, relu))
network.add(Layer(16, 16, relu))
network.add(Layer(16, 16, relu))
network.add(Layer(16, 1, linear))

def visualize_network(input_size, num_hidden_layers, hidden_size, output_size):
    screen.fill((0,0,0))

    side_offset = 40
    hor_gap = (WINDOW_WIDTH - 2 * side_offset) / (num_hidden_layers + 1)
    node_gap = 36
    node_radius = 12
    for i in range(input_size):
        y = WINDOW_HEIGHT / 2 - node_gap * (input_size / 2) + i * node_gap
        pygame.draw.circle(screen, (255,255,255), (side_offset, y), node_radius, 3)
    for q in range(num_hidden_layers):
        for i in range(hidden_size):
            y = WINDOW_HEIGHT / 2 - node_gap * (hidden_size / 2) + i * node_gap
            pygame.draw.circle(screen, (255,255,255), (side_offset + hor_gap * (q + 1), y), node_radius, 3)
    for i in range(output_size):
        y = WINDOW_HEIGHT / 2 - node_gap * (output_size / 2) + i * node_gap
        pygame.draw.circle(screen, (255,255,255), (side_offset + (num_hidden_layers + 1) * hor_gap, y), node_radius, 3)

        for q in range(num_hidden_layers):
            if q == 0:
                start_x = side_offset
                end_x = side_offset + hor_gap
                for i in range(input_size):
                    start_y = WINDOW_HEIGHT / 2 - node_gap * (input_size / 2) + i * node_gap
                    for p in range(hidden_size):
                        weight = network.layers[0].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = WINDOW_HEIGHT / 2 - node_gap * (hidden_size / 2) + p * node_gap
                        pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))
            else:
                start_x = side_offset + q * hor_gap
                end_x = side_offset + (q + 1) * hor_gap
                for i in range(hidden_size):
                    start_y = WINDOW_HEIGHT / 2 - node_gap * (hidden_size / 2) + i * node_gap
                    for p in range(hidden_size):
                        weight = network.layers[q + 1].weights[i, p]
                        if weight > 0:
                            color = (255, 255, 255)
                        else:
                            color = (0, 134, 212)
                        end_y = WINDOW_HEIGHT / 2 - node_gap * (hidden_size / 2) + p * node_gap
                        pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))

        for q in range(hidden_size):
            start_x = side_offset + hor_gap * (num_hidden_layers)
            end_x = side_offset + hor_gap * (num_hidden_layers + 1)
            start_y = WINDOW_HEIGHT / 2 - node_gap * (hidden_size / 2) + q * node_gap
            for p in range(output_size):
                weight = network.layers[num_hidden_layers].weights[q, p]
                if weight > 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 134, 212)
                end_y = WINDOW_HEIGHT / 2 - node_gap * (output_size / 2) + p * node_gap
                pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), max(1, int(abs(weight) * 7)))

    pygame.display.update()

def run_training(epochs, lr):
    x = np.linspace(-200, 200, 1000).reshape(-1, 1)
    correct_y = x // 200

    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = correct_y.mean(), correct_y.std()

    x_norm = (x - x_mean) / x_std
    y_norm = (correct_y - y_mean) / y_std

    for epoch in range(epochs):
        predicted_y_norm = network.forward(x_norm)
        loss = mse(predicted_y_norm, y_norm)
        network.backward(predicted_y_norm, y_norm)
        network.update(lr)

run_training(25000, 0.05)