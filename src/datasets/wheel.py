import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
import io

class WheelDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        #load the data using the torch.load function
        #The pt files will search for images (./sub_normal_data/Processed/) and annotations (sub_normal_data_annotations.csv) in the ./Wheel/ folder inside of your master directory
        training_data = torch.load('./data/Wheel/sub_training_set.pt')
        testing_data = torch.load('./data/Wheel/sub_testing_set.pt')
        anomaly_data = torch.load('./data/Wheel/anomalous_spokes.pt') #test on two anomaly sets: anomalous spokes and anomalous hubs

        self.train_set = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
        self.test_set = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True)
        self.anomaly_set = DataLoader(dataset=anomaly_data, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return(image, y_label)