from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import shutil
import torch.utils.data as data
import torchvision.transforms as transforms


class ICGStuff(data.Dataset):
    N_CLASSES = 9
    IMG_HEIGHT = 426
    IMG_WIDTH = 640

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
        seg = transforms.ToTensor()(seg)

        return img, seg

    def __len__(self):
        return len(self.img_names)


class ICGStuffBuilder:
    IMG_HEIGHT = 426
    IMG_WIDTH = 640
    CLASSES_TO_TURN_OFF = {
        "bicycle": [119, 11, 32],
        "dirt": [130, 76, 0],
        "gravel": [112, 103, 87],
        "water": [28, 42, 168],
        "fence-pole": [153, 153, 153],
        "person": [255, 22, 96],
        "dog": [102, 51, 0],
        "bald-tree": [190, 250, 190],
        "air-marker": [112, 150, 146],
        "conflicting": [255, 0, 0],
        "door": [254, 148, 12],
        "window": [254, 228, 12],
        "fence": [190, 153, 153],
        "tree": [51, 51, 0],
        "obstacle": [2, 135, 115]
    }
    GRASS_COLOR = [0, 102, 0]

    def build(self, root):
        self.root = root
        self.image_folder = Path(self.root, "images")
        self.target_folder = Path(self.root, "gt", "semantic", "label_images")
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
                               trainset_folder, 0.9)
            self._create_split(resized_image_folder, processed_target_folder,
                               valset_folder, 0.1)
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

            if len(train_images) < 360 or len(val_images) < 40:
                raise ValueError("Dataset corrupted. ")

        return trainset_folder, valset_folder

    def _create_split(self, resized_image_folder, processed_target_folder,
                      folder, size):
        resized_image_names = [
            f for f in os.listdir(resized_image_folder)
            if os.path.isfile(Path(resized_image_folder, f))
        ]
        train_image_folder = Path(folder, "images")
        os.makedirs(train_image_folder)
        for resized_image_name in resized_image_names[:(
                len(resized_image_names) * size)]:
            shutil.move(
                Path(resized_image_folder, resized_image_name),
                Path(train_image_folder, resized_image_name))

        processed_target_names = [
            f for f in os.listdir(processed_target_folder)
            if os.path.isfile(Path(processed_target_folder, f))
        ]
        train_target_folder = Path(folder, "targets")
        os.makedirs(train_image_folder)
        for processed_target_name in processed_target_names[:(
                len(processed_target_names) * size)]:
            shutil.move(
                Path(processed_target_folder, processed_target_name),
                Path(train_target_folder, processed_target_name))

    def _resize_images(self):
        resized_image_folder = Path(self.root, "images_resized")
        os.makedirs(resized_image_folder)
        print("Resizing images...")
        for filename in tqdm(os.listdir(self.image_folder)):
            filepath = Path(self.image_folder, filename)
            if os.path.isfile(filepath):
                Image.open(filepath).resize(
                    (self.IMG_WIDTH, self.IMG_HEIGHT)).save(
                        Path(resized_image_folder, filename))
        return resized_image_folder

    def _process_targets(self):
        processed_target_folder = Path(self.root, "gt_proc")
        print("Processing ground truth masks...")
        for filename in tqdm(os.listdir(self.target_folder)):
            filepath = Path(self.target_folder, filename)
            if os.path.isfile(filepath):

                dest_path = Path(processed_target_folder, filename)
                image = Image.open(filepath)
                img_array = np.array(image)

                # Turn off classes
                img_array[np.where(
                    np.logical_or.reduce(
                        (img_array == self.CLASSES_TO_TURN_OFF['door'],
                         img_array == self.CLASSES_TO_TURN_OFF['bicycle'],
                         img_array == self.CLASSES_TO_TURN_OFF['dirt'],
                         img_array == self.CLASSES_TO_TURN_OFF['gravel'],
                         img_array == self.CLASSES_TO_TURN_OFF['water'],
                         img_array == self.CLASSES_TO_TURN_OFF['fence-pole'],
                         img_array == self.CLASSES_TO_TURN_OFF['person'],
                         img_array == self.CLASSES_TO_TURN_OFF['dog'],
                         img_array == self.CLASSES_TO_TURN_OFF['bald-tree'],
                         img_array == self.CLASSES_TO_TURN_OFF['window'],
                         img_array == self.CLASSES_TO_TURN_OFF['air-marker'],
                         img_array == self.CLASSES_TO_TURN_OFF['conflicting']
                         )).all(axis=2))] = [0, 0, 0]

                # Change grass color
                img_array[np.where(
                    (img_array == self.GRASS_COLOR).all(axis=2))] = [
                        126, 126, 128
                    ]

                img_proc = Image.fromarray(img_array)
                img_proc.resize((self.IMG_WIDTH,
                                 self.IMG_HEIGHT)).save(dest_path)
        return processed_target_folder
