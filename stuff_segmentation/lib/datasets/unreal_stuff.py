from PIL import Image
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class UnrealStuff(data.Dataset):
    N_CLASSES = 8
    IMG_HEIGHT = 426
    IMG_WIDTH = 640
    CLASSES = {
        (128, 64, 128): "paved-area",
        (48, 41, 30): "rocks",
        (0, 50, 89): "pool",
        (28, 42, 168): "water",
        (107, 142, 35): "vegetation",
        (70, 70, 70): "roof",
        (102, 102, 156): "wall",
        (0, 102, 0): "grass"
    }
    CLASS_INDEXES = {
        "paved-area": 1,
        "rocks": 2,
        "pool": 3,
        "water": 4,
        "vegetation": 5,
        "roof": 6,
        "wall": 7,
        "grass": 8
    }

    def __init__(self, root):
        self.root = root
        image_folder = Path(self.root, "images")
        self.img_names = [
            f for f in os.listdir(image_folder)
            if os.path.isfile(Path(image_folder, f))
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the
                   image.
        """
        img_name = self.img_names[index]
        img_path = Path(self.root, "images", img_name)
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)

        seg_name = img_name.replace(".jpg", ".png")
        seg_path = Path(self.root, "targets", seg_name)
        seg = Image.open(seg_path)
        seg_array = np.array(seg)

        seg = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH))
        for i in range(self.IMG_HEIGHT):
            for j in range(self.IMG_WIDTH):
                if tuple(seg_array[i, j]) in self.CLASSES.keys():
                    class_ = self.CLASSES[tuple(seg_array[i, j])]
                    seg[i, j] = self.CLASS_INDEXES[class_]
        seg = torch.from_numpy(seg)

        return img, seg

    def __len__(self):
        return len(self.img_names)


class UnrealStuffBuilder:
    IMG_HEIGHT = 426
    IMG_WIDTH = 640
    NUM_TRAIN_IMAGES = 360
    NUM_VAL_IMAGES = 40

    def build(self, root):
        self.root = root
        self.image_folder = Path(self.root, "images")
        self.target_folder = Path(self.root, "masks")
        if ((not os.path.exists(self.image_folder))
                or (not os.path.exists(self.target_folder))):
            raise RuntimeError("Dataset not found.")

        trainset_folder = Path(self.root, "train")
        valset_folder = Path(self.root, "val")
        if ((not os.path.exists(trainset_folder))
                or (not os.path.exists(valset_folder))):
            resized_image_folder = self._resize_images()
            processed_target_folder = self._process_targets()
            self._create_split(resized_image_folder, processed_target_folder,
                               trainset_folder, self.NUM_TRAIN_IMAGES)
            self._create_split(resized_image_folder, processed_target_folder,
                               valset_folder, self.NUM_VAL_IMAGES)
        else:
            train_image_folder = Path(trainset_folder, "images")
            train_images = [
                f for f in os.listdir(train_image_folder)
                if os.path.isfile(Path(train_image_folder, f))
            ]

            val_image_folder = Path(valset_folder, "images")
            val_images = [
                f for f in os.listdir(val_image_folder)
                if os.path.isfile(Path(val_image_folder, f))
            ]

            if ((len(train_images) < self.NUM_TRAIN_IMAGES)
                    or (len(val_images) < self.NUM_VAL_IMAGES)):
                raise ValueError("Dataset corrupted. ")

        return trainset_folder, valset_folder

        raise NotImplementedError
