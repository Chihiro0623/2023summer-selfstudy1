import os
import pickle
from typing import Optional, Callable

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset, CIFAR100
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.transforms import transforms


class CIFAR100V2(VisionDataset):
    """Extension of CIFAR100 dataset of torchvision.
    This contains the distorted dataset also.
    You just replace the train=[True|False] to dtype=[train|test|distort]
    Distort provides only image data, and the y labels are all -1.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]
    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    distort_list = [
        ["distort", "-"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            root: str,
            ctype: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            intensity = None
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.dtype = ctype

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.dtype == 'train':
            downloaded_list = self.train_list
        elif self.dtype == 'test':
            downloaded_list = self.test_list
        elif self.dtype == 'distort':
            downloaded_list = self.distort_list
        else:
            raise AttributeError(f'CIFAR100V2 dtype is not supported {self.dtype}. '
                                 f'Only dtype=[train|test|distort] is allowed.')

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                elif "fine_labels" in entry:
                    self.targets.extend(entry["fine_labels"])
                if "distorted_type" in entry:
                    self.distorted_type = entry["distorted_type"]

        if self.dtype in ['train', 'test']:
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            self.data = self.data[0]
            # self.targets = [-1] * self.data.shape[0]
        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


if __name__ == '__main__':
    root = '~/shared/hdd_ext/nvme1/classification'
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Original torchvision CIFAR100 Dataset
    train_ds = CIFAR100(root, train=True, transform=transform)
    valid_ds = CIFAR100(root, train=False, transform=transform)

    # CIFAR100V2 for distort dataset
    train_ds = CIFAR100V2(root, ctype='train', transform=transform)
    valid_ds = CIFAR100V2(root, ctype='test', transform=transform)
    dist_ds = CIFAR100V2(root, ctype='distort', transform=transform)

    # For check the images
    from matplotlib import pyplot as plt

    for i in range(5):
        x, y = dist_ds.__getitem__(i)
        print(x.shape, y)

        x = x.mul(255).byte()
        plt.imshow(Image.fromarray(x.permute(1, 2, 0).cpu().numpy(), mode='RGB'))
        plt.show()
