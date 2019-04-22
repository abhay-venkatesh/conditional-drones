import sys
import argparse
import numpy as np

from os import walk
import imgaug as ia
from scipy.misc import imsave
from skimage.io import imread
from skimage.measure import label, regionprops
from pascal_voc_writer import Writer

# define detectable classes and corresponding colors
# tuple keys are (R, G, B)
# RGB_classes = {(128, 64, 128): 'paved-area',
# 			   (48, 41, 30): 'rocks',
# 			   (0, 50, 89): 'pool',
# 			   (28, 42, 168): 'water',
# 			   (107, 142, 35): 'vegetation',
# 			   (70, 70, 70): 'roof',
# 			   (102, 102, 156): 'wall',
# 			   (190, 153, 153): 'fence',
# 			   (9, 143, 150): 'car',
# 			   (51, 51, 0): 'tree',
# 			   (2, 135, 115): 'obstacle',
# 			   (0, 102, 0): 'grass' }
RGB_classes = {(9, 143, 150): 'car'}

box_width = 2

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mask_dir',
    type=str,
    default=None,
    required=True,
    help='segmentation mask image directory')
parser.add_argument(
    '--image_dir',
    type=str,
    default=None,
    required=True,
    help='real image directory')
parser.add_argument(
    '--out_dir',
    type=str,
    default=None,
    required=True,
    help='annotation output directory')
parser.add_argument(
    '--bbox_dir',
    type=str,
    default=None,
    required=True,
    help='bounding box annoted image output directory')
args = parser.parse_args()

mask_dir = args.mask_dir
image_dir = args.image_dir
out_dir = args.out_dir
bbox_dir = args.bbox_dir

if mask_dir == out_dir or image_dir == out_dir:
    print('input and output dirs cannot be the same')
    exit()

files = []
for (dirpath, dirnames, filenames) in walk(mask_dir):
    files.extend(filenames)
    break
if not files:
    print('no images found in input directory')
    exit()

for f in files:
    print('annotating %s...' % f)
    img_id = f[:-4]  # strip '.png'
    mask_img_path = '%s/%s' % (mask_dir, f)
    real_img_path = '%s/%s.jpg' % (image_dir, img_id)

    # read in mask and real images
    mask_img = imread(mask_img_path, img_num=0)
    real_img = imread(real_img_path, img_num=0)

    img_w = mask_img.shape[1]  # number of columns is 'width'
    img_h = mask_img.shape[0]  # number of rows is 'height'
    img_size = img_w * img_h

    # Pascal VOC Annotations
    voc_writer = Writer(real_img_path, img_w, img_h)

    label_img = label(mask_img)

    # each connected-component will have its own label
    # it will repeat 3 times for RGB so only use every 3rd
    seen = {}
    for region in regionprops(label_img):

        # get bounding box for region
        # format: (min_row, min_col, ~, max_row, max_col, ~)
        bbox = region.bbox
        min_row = bbox[0]
        min_col = bbox[1]
        max_row = bbox[3]
        max_col = bbox[4]
        bbox_area = (max_row - min_row) * (max_col - min_col)

        # ignore regions whose area > 50% of image
        max_area = img_size / 2
        if region.area > max_area or bbox_area > max_area:
            continue

        # ignore small regions and artifcats, < 0.1% of image
        min_area = 0.001 * img_size
        if region.area < min_area or bbox_area < min_area:
            continue

        # find centroid coords, only need row and col
        centroid = np.rint(region.centroid[0:2]).astype(np.int32)

        # get corresponding RGB color and label
        RGB_key = tuple(mask_img[centroid[0], centroid[1]])

        if RGB_key not in RGB_classes:
            continue  # we don't care about this label
        if (bbox[1], bbox[0], bbox[4], bbox[3]) in seen:
            continue  # we've seen this bounding box

        class_label = RGB_classes[RGB_key]
        seen[(bbox[1], bbox[0], bbox[4], bbox[3])] = True

        #print('%s - area: %d' % (class_label, region.area))
        #print(centroid)
        #print('xmin: %d\nymin:%d\nxmax: %d\nymax: %d\n' % (min_col, min_row, max_col, max_row))

        # add this bounding box to our annotation
        # format: name, xmin, ymin, xmax, ymax
        voc_writer.addObject(class_label, min_col, min_row, max_col, max_row)

        # draw actual bounding box
        box_color = mask_img[centroid[0], centroid[1]]
        bb = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=min_col, x2=max_col, y1=min_row, y2=max_row)],
            shape=real_img.shape)
        real_img = bb.draw_on_image(
            real_img, color=box_color, thickness=box_width)

    # save annotations in VOC xml format
    voc_writer.save('%s/%s.xml' % (out_dir, img_id))

    # save box annotations of real img to out directory
    imsave('%s/%s.jpg' % (bbox_dir, img_id), real_img)
