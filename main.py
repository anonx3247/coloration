from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import pandas as pd
from data import *
from model import Model
import kaggle

DATA_DIR = "dataset"
CSV = "images.csv"

create_csv(path=DATA_DIR, csv_path=CSV)

BATCH_SIZE = 64
LATENT_DIM = 100

training_data = TrainDataset(img_csv=CSV, img_dir=DATA_DIR)

# On charge notre dataset dans un dataloader pour le batcher et le mélanger
x_train = DataLoader(training_data, batch_size = BATCH_SIZE, shuffle = True)
