from PIL import Image
from pathlib import Path
import os
import torch.utils.data as data
import torchvision.transforms as transforms


class TranslatedUnreal(data.Dataset):
    IMG_HEIGHT = 426
    IMG_WIDTH = 640

    def __init__(self, root):
        self.root = root
        image_folder = Path(self.root)
        self.img_names = [
            f for f in os.listdir(image_folder)
            if os.path.isfile(Path(image_folder, f))
        ]

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = Path(self.root, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
        img = transforms.ToTensor()(img)
        return img

    def __len__(self):
        return len(self.img_names)