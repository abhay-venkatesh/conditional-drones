from pathlib import Path
import sys
import numpy as np

MANIFEST_FILE = sys.argv[1]
TRAIN_FILE = sys.argv[2]
TEST_FILE = sys.argv[3]

with open(MANIFEST_FILE) as f:
    imgs = [x.strip() for x in f.readlines()]

# shuffle data
np.random.shuffle(imgs)

# split 80/20 for train/test
train_split = len(imgs) * 0.8
train, test = imgs[:train_split], imgs[train_split:]

# write to files
np.savetxt(TRAIN_FILE, train, delimiter='\n')
np.savetxt(TEST_FILE, test, delimiter='\n')