import numpy as np
import torch
from torch import nn

def create_model():
    # Linear layer mapping from 784 features, so it should be 784->256->16->10
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )

    return model

def count_parameters(model):
    # Counting the total number of parameters in the model
    return sum(p.numel() for p in model.parameters())
