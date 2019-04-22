import torch.utils.data as data
from pathlib import Path
import os
import torchvision.transforms as transforms
from PIL import Image


class ICG(data.Dataset):
    N_CLASSES = 13

    def __init__(self, root):
        self.imagefolder = Path(root, "images")
        self.targetfolder = Path(root, "targets")
        self.images = [
            f for f in os.listdir(self.imagefolder)
            if os.path.isfile(Path(self.imagefolder, f))
        ]
        self.targets = [
            f for f in os.listdir(self.targetfolder)
            if os.path.isfile(Path(self.targetfolder, f))
        ]

    def __getitem__(self, index):
        img = Image.open(Path(self.imagefolder, self.images[index]))
        target = Image.open(Path(self.targetfolder, self.targets[index]))
        return transforms.ToTensor()(img), transforms.ToTensor()(target)

    def __len__(self):
        return len(self.images)
