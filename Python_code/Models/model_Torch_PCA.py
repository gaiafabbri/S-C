import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


'''------------------ MODEL DEFINITION ------------------'''
'''The model is defined as follows:
1) The Reshape layer is necessary to make the input match the number of principal componests desired 
2) Three Linear layers are present, with 64, 32 and 64 neurons respectively
3)The relu activation function after each convolutional layer is not linear and helps to learn complex characteristics in the model
4) The Sigmoid layer is introduced to have an aoutput between 0 and 1 which is interpreted as the probability to belong to signal or background classes
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
 
