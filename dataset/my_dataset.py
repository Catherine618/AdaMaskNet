import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


    @staticmethod
    def collate_fn(batch):
        data, labels = zip(*batch)
        data = torch.stack([torch.tensor(d) for d in data], 0)
        labels = torch.tensor(labels)
        return data, labels

