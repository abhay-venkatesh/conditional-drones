from PIL import Image
from pathlib import Path
from skimage.transform import resize
from tqdm import tqdm
import csv
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class ICGStuff(data.Dataset):
    N_CLASSES = 2
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

    def __init__(self, root):
        raise NotImplementedError
        self.root = root
        self._build()

        self.img_names = []
        self.bbox_indexes = []

        ann_file_path = Path(self.root, "annotations.csv")
        with open(ann_file_path, newline='') as ann_file:
            reader = csv.reader(ann_file, delimiter=',')
            for row in reader:
                self.img_names.append(row[0])
                self.bbox_indexes.append(row[1])

    def _build(self):
        self.image_folder = Path(self.root, "images")
        self.target_folder = Path(self.root, "gt", "semantic", "label_images")
        if ((not os.path.exists(self.image_folder))
                or (not os.path.exists(self.target_folder))):
            raise RuntimeError("Dataset not found.")

        self.trainset_folder = Path(self.root, "train")
        self.valset_folder = Path(self.root, "val")
        if ((not os.path.exists(self.trainset_folder))
                or (not os.path.exists(self.valset_folder))):
            os.makedirs(self.trainset_folder)
            os.makedirs(self.valset_folder)

            # Resize the giganormous images
            resized_image_folder = Path(self.root, "images_resized")
            for filename in os.listdir(self.image_folder):
                filepath = Path(self.image_folder, filename)
                if os.path.isfile(filepath):
                    Image.open(filepath).resize(
                        self.IMG_WIDTH, self.IMG_HEIGHT).save(
                            Path(resized_image_folder, filename))

            # Process the ground truth semantic segmentation
            processed_target_folder = Path(self.root, "gt_proc")
            self._process_targets(processed_target_folder)

        else:
            train_image_folder = Path(self.trainset_folder, "images")
            train_images = [
                f for f in os.listdir(train_image_folder)
                if os.path.isfile(Path(train_image_folder, f))
            ]
            val_image_folder = Path(self.valset_folder, "images")
            val_images = [
                f for f in os.listdir(val_image_folder)
                if os.path.isfile(Path(val_image_folder, f))
            ]
            if len(train_images) < 360 or len(val_images) < 40:
                raise ValueError("Dataset corrupted. ")

    def _process_targets(self, processed_target_folder):
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
                img_proc.save(dest_path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the
                   image.
        """
        raise NotImplementedError
        img_name = self.img_names[index]
        img_path = Path(self.root, "images", img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT), Image.ANTIALIAS)
        img = transforms.ToTensor()(img)

        seg_name = img_name.replace(".jpg", ".png")
        seg_path = Path(self.root, "annotations", seg_name)
        seg = Image.open(seg_path)
        S = np.array(seg)
        S = resize(
            S, (self.IMG_HEIGHT, self.IMG_WIDTH),
            anti_aliasing=False,
            mode='constant')
        S = np.where(S > 0, 1, 0)
        seg = torch.from_numpy(S)

        return img, seg

    def __len__(self):
        return len(self.img_names)
