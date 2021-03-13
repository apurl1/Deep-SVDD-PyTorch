import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from .utils import WheelDataset

class WheelDatasets(Dataset):
    def __init__(self, class_id, transform=None):
        batch_size = 32

        if class_id == 0:
            anomaly_data = torch.load('../data/Wheel/anomalous_hubs.pt')
            self.anomaly_set = DataLoader(dataset=anomaly_data, batch_size=batch_size, shuffle=True)
        elif class_id == 1:
            anomaly_data = torch.load('../data/Wheel/anomalous_spokes.pt')
            self.anomaly_set = DataLoader(dataset=anomaly_data, batch_size=batch_size, shuffle=True)
        
        train_data = torch.load('../data/Wheel/sub_training_set.pt')
        self.train_set = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_data = torch.load('../data/Wheel/sub_testing_set.pt')
        self.test_set = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)