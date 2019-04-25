import sys
import argparse
import numpy as np

from os import walk
from scipy.misc import imsave
from skimage.io import imread

# define detectable classes and corresponding colors
# tuple keys are (R, G, B)
RGB_classes = {(128, 64, 128): 'paved-area',
			   (48, 41, 30): 'rocks',
			   (0, 50, 89): 'pool',
			   (28, 42, 168): 'water',
			   (107, 142, 35): 'vegetation',
			   (70, 70, 70): 'roof',
			   (102, 102, 156): 'wall',
			   (190, 153, 153): 'fence',
			   (9, 143, 150): 'car',
			   (51, 51, 0): 'tree',
			   (2, 135, 115): 'obstacle',
			   (0, 102, 0): 'grass' }


parser = argparse.ArgumentParser()
parser.add_argument(
    '--mask_dir',
    type=str,
    default=None,
    required=True,
    help='segmentation mask image directory')
parser.add_argument(
    '--out_dir',
    type=str,
    default=None,
    required=True,
    help='annotation output directory')

args = parser.parse_args()

mask_dir = args.mask_dir
out_dir = args.out_dir

TOL = 10 # tolerance for matching mask

files = []
for (dirpath, dirnames, filenames) in walk(mask_dir):
    files.extend(filenames)
    break
if not files:
    print('no images found in input directory')
    exit()

for f in files:
    print('smoothing %s...' % f)
    img_id = f[:-4]  # strip '.png'
    img_path = '%s/%s' % (mask_dir, f)

    # read in mask and real images
    img = imread(img_path, img_num=0)

    img_w = img.shape[1]  # number of columns is 'width'
    img_h = img.shape[0]  # number of rows is 'height'

    for i in range(img_h):
        for j in range(img_w):

            # iterate over class colors
            match = False
            for RGB_key in RGB_classes.keys():
                if  (img[i, j, 0] >= RGB_key[0] - TOL) \
                and (img[i, j, 0] <= RGB_key[0] + TOL) \
                and (img[i, j, 1] >= RGB_key[1] - TOL) \
                and (img[i, j, 1] <= RGB_key[1] + TOL) \
                and (img[i, j, 2] >= RGB_key[2] - TOL) \
                and (img[i, j, 2] <= RGB_key[2] + TOL):
                    match = True
                    break

            if match:
                img[i, j, :] = RGB_key

    # save box annotations of real img to out directory
    imsave('%s/%s' % (out_dir, f), img)
