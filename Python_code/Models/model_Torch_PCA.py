import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


'''------------------ MODEL DEFINITION ------------------'''
'''The model is defined as follows:
1) The Reshape layer is necessary since we are dealing with images presented as multidimensional arrays
2) The Conv2d layer extracts the images pattern ("padding"=1 means that borders are not reduced)
3)The relu activation function after each convolutional layer is not linear and helps to learn complex characteristics in the model
4) The BatchNormalization layer is used to make the training more reliable and faster
5) The MaxPooling layer reduces the input spatial dimension
6) The Flatten layer ensures that the output of the previous steps is a one-dimensional array to be passed to the following dense layer with a relu activation function
7) The linear layers are used to covnert the extracted features into a binary classification suitable for this analysis
6)the output layer uses a sigmoid function to generate the probability distribution for an image to belong to the singal or background classes; this is common for binary classification problem
'''


def torch_PCA(width, height, num_components_signal):  # Aggiunto num_components_signal come parametro
    # Custom Reshape Layer
    class Reshape(torch.nn.Module):
        def forward(self, x):
            return x.view(-1, num_components_signal)  # Modificato per utilizzare num_components_signal

    # Modello completamente connesso
    model = torch.nn.Sequential(
        Reshape(),
        nn.Linear(num_components_signal, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    return model  # Modificato per restituire net invece di model
 
