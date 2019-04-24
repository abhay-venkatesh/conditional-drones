from pathlib import Path
import os
import shutil

TRAIN_FILE = "unreal_splits/4000/train.split"
TEST_FILE = "unreal_splits/4000/test.split"
IMG_DIR = Path("/mnt/c/Users/viswe/Desktop/computer-science/CS 766/unreal_images_4000/images")
EXTENSION = "-outputs.png"

with open(TRAIN_FILE) as f:
    train_imgs = [x.strip() + EXTENSION for x in f.readlines()]

with open(TEST_FILE) as f:
    test_imgs = [x.strip() + EXTENSION for x in f.readlines()]

print(os.listdir(IMG_DIR))

train_path = Path(IMG_DIR, "train")
if not os.path.exists(train_path):
    os.mkdir(train_path)
for t in train_imgs:
    print(t)
    shutil.move(Path(IMG_DIR, t), Path(train_path, t))

test_path = Path(IMG_DIR, "val")
if not os.path.exists(test_path):
    os.mkdir(test_path)

for t in test_imgs:
    shutil.move(Path(IMG_DIR, t), Path(test_path, t))
