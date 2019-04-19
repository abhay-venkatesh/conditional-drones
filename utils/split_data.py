import os
import sys
import shutil

train_file = sys.argv[1]
test_file = sys.argv[2]
img_dir = sys.argv[3]

with open(train_file) as f:
	train_imgs = [x.strip() + '.png' for x in f.readlines()]

with open(test_file) as f:
	test_imgs = [x.strip() + '.png' for x in f.readlines()]

print(os.listdir(img_dir))

for t in train_imgs:
	print(t)
	shutil.move(img_dir + '/' + t, 'train_labels/' + t)

for t in test_imgs:
	print(t)
	shutil.move(img_dir + '/' + t, 'test_labels/' + t)
