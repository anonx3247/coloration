from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
import numpy as np
import pandas as pd
import os
import pandas as pd


def create_csv(path, csv_path):       
    csv = open(csv_path, 'w')
    for file in os.listdir(path):
        csv.write(f"{file}\n")

    csv.close()

BATCH_SIZE = 64
LATENT_DIM = 100

#On crée un dataset custom à partir des images du dossier 'dataset'
class TrainDataset(Dataset):
    def __init__(self, img_csv, img_dir):
        self.img_dir = img_dir
        self.img_csv = pd.read_csv(img_csv)
        self.resize = transforms.Resize((64, 64))
        self.normalize = lambda t : t * 2 / 255 -1

    def __len__(self):
        return len(self.img_csv)

    def __getitem__(self, index):
        img = read_image(self.img_dir + "/" + self.img_csv[index], 'r')
        img = self.normalize(img)
        return self.resize(img)


# On charge notre dataset dans un dataloader pour le batcher et le mélanger
x_train = DataLoader(training_data, batch_size = BATCH_SIZE, shuffle = True)
