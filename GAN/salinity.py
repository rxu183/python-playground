import numpy as np
import pandas as pd
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#Steps for Performing even a Linear Regression:
#1. Read in/Clean the data:
#2. Create the model: In this case, we're using L
def 