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

def center_drawing(pixelmatrix):
    pixelmatrix_2d = pixelmatrix.reshape(28, 28)
    total_mass = np.sum(pixelmatrix_2d)
    y_indices, x_indices = np.ogrid[:28, :28]
    center_gravity_x = np.sum(pixelmatrix_2d * x_indices) / total_mass
    center_gravity_y = np.sum(pixelmatrix_2d * y_indices) / total_mass
    x_gravity_offset = -round(center_gravity_x - 14)
    y_gravity_offset = -round(center_gravity_y - 14)
    centered_2d_matrix = np.roll(pixelmatrix_2d, shift=(y_gravity_offset, x_gravity_offset), axis=(0, 1))
    if y_gravity_offset > 0:
        centered_2d_matrix[:y_gravity_offset, :] = 0.0
    elif y_gravity_offset < 0:
        centered_2d_matrix[y_gravity_offset:, :] = 0.0
    if x_gravity_offset > 0:
        centered_2d_matrix[:, :x_gravity_offset] = 0.0
    elif x_gravity_offset < 0:
        centered_2d_matrix[:, x_gravity_offset:] = 0.0
    return centered_2d_matrix.flatten()

def cords_to_array(cords_input):
    x = cords_input[0]
    y = cords_input[1]
    return y * 28 + x + 1

def array_to_cords(number_input):
    y = np.floor(number_input / 28)
    x = number_input - y * 28
    return (x,y)

def softmax(output_array):
    shift_array = output_array - np.max(output_array, axis=-1, keepdims=True)
    exps = np.exp(shift_array)
    return exps / np.sum(exps, axis=-1, keepdims=True)
