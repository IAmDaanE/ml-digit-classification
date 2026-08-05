import numpy as np

def load_mnist_csv(file_path):    
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    labels = data[:, 0].astype(int)
    pixels = data[:, 1:] / 255.0
    num_samples = labels.shape[0]
    one_hot_labels = np.zeros((num_samples, 10))
    one_hot_labels[np.arange(num_samples), labels] = 1.0
    return pixels, one_hot_labels

def shuffle_dataset(images, labels):
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
    new_images = images[indices]
    new_labels = labels[indices]
    return new_images, new_labels