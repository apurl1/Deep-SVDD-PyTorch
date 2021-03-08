import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchsummary import summary
import io

class WheelDataset(Dataset):
    def __init__(self, csvfile, root_dir, ptfile, transform=None):
        self.annotations = pd.read_csv(csvfile)
        self.root_dir = root_dir
        self.transform = transform

        #load the data using the torch.load function
        #The pt files will search for images (./sub_normal_data/Processed/) 
        #and annotations (sub_normal_data_annotations.csv) in the ./Wheel/ folder inside of your master directory
        data = torch.load(ptfile)
        batch_size = 32
        self.dataset = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return(image, y_label)

class WheelDatasets():
    def __init__(self, class_id, transform=None):
        if class_id == 0:
            self.anomaly_set = WheelDataset(csvfile='../Wheel/anomalous_hubs_annotations.csv', 
                                            root_dir='../Wheel/anomalous_hubs/Processed', 
                                            ptfile='../data/Wheel/anomalous_hubs.pt')
        elif class_id == 1:
            self.anomaly_set = WheelDataset(csvfile='../Wheel/anomalous_spokes_annotations.csv', 
                                root_dir='../Wheel/anomalous_spokes/Processed', 
                                ptfile='../data/Wheel/anomalous_spokes.pt')
        self.train_set = WheelDataset(csvfile='../Wheel/sub_normal_data_annotations.csv', 
                                root_dir='../Wheel/sub_normal_data/Processed', 
                                ptfile='../data/Wheel/sub_training_set.pt')
        self.test_set = WheelDataset(csvfile='../Wheel/sub_normal_data_annotations.csv', 
                                root_dir='../Wheel/sub_normal_data/Processed', 
                                ptfile='../data/Wheel/sub_testing_set.pt')
        combined_test_data = [self.test_set, self.anomaly_set]
        self.test_set = ConcatDataset(combined_test_data)