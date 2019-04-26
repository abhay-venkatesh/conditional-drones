from pathlib import Path
import os
import sys
import shutil

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
IMG_DIR = Path(sys.argv[3])
EXTENSION = ".jpg"

with open(TRAIN_FILE) as f:
    train_imgs = [x.strip() + EXTENSION for x in f.readlines()]

with open(TEST_FILE) as f:
    test_imgs = [x.strip() + EXTENSION for x in f.readlines()]

print(os.listdir(IMG_DIR))

train_path = IMG_DIR#Path(IMG_DIR, "train")
if not os.path.exists(train_path):
    os.mkdir(train_path)
for t in train_imgs:
    print(t)
    shutil.move(Path(IMG_DIR, t), Path(train_path, t))

test_path = Path(IMG_DIR, "../test_boxes")
if not os.path.exists(test_path):
    os.mkdir(test_path)

for t in test_imgs:
    shutil.move(Path(IMG_DIR, t), Path(test_path, t))
