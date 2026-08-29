# MNIST Digit Recognition

A neural network algorithm that recognizes and classifies hand drawn digits. Written in python using my very own [neural network machine learning library](https://github.com/IAmDaanE/neural-network-library).

## About the Project

This repo contains a python file to train the image classification algorithm, two pretrained models and an environment where you can draw a digit and see what the model thinks it is. I used the freely available MNIST dataset which contains 70.000 labeled greyscale images (60.000 for training and 10.000 for testing). To make it easier to load in and download i used a CSV format one [from Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). 

The neural network for my models has 3 hidden layers with a size 128 nodes and uses relu activation for the hidden layers and softmax for the output layer, which makes all outputs positive and boosts the big confident ones. The loss function is cross entropy loss which is usually implemented with softmax.

## Project Structure

```
ml-digit-classification/
├── data/                   # contains the mnist dataset                                       
    └── ... 
├── models/                 # models with pure numpy arrays for the weights and biases           
    └── ...       
└── src/                
    ├── accuracy_tester.py  # test what score a model got on the test dataset
    ├── predict_drawing.py  # the environment where you can draw something yourself
    ├── train.py            # the actual training loop
    └── utils.py            # some other reusable functions
```

## Getting Started

### Installing the Project

**Requirements**: You must have Python 3.10 - 3.13.

1. Clone the repository or download the zip and unpack it to your directory of choice.
2. Navigate to that directory in a terminal.
3. In a venv or the global python version install the needed libraries.
    ```
    pip install -r requirements.txt
    ```

### Running the Drawing Environment

In the project directory:
```
python src/predict_drawing.py
```

### Training Your Own Model

The actual dataset isnt in this repo because its way to big for a github repository but you can download mnist_train.csv and mnist_test.csv [from this kaggle page](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). Then configure your wanted parameters in train.py. When you are happy with the amount of loss during training press Ctrl + C to quit training and save the weights and biases.

## License

This project is open-source and available under the MIT License.