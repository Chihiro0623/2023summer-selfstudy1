import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pickle

def unpickle(path):
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CustomDataset(Dataset):
    def __init__(self, path, train=True, transform=None, target_transform=None):
        self.path = path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.train_data = []
            self.train_labels = []

            entry = unpickle(self.path)
            self.train_data.append(entry[b'data'])
            self.train_labels += entry[b'fine_labels']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            entry = unpickle(self.path)
            self.test_data = entry[b'data']
            self.test_labels = entry[b'fine_labels']

            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


