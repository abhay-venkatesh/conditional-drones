from PIL import Image
import numpy as np
import argparse
import os

import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir",
    required=True,
    help="path to folder containing segmented images")
parser.add_argument("--output_dir", required=True, help="output path")

args = parser.parse_args()

seg_class = {'obstacle': [2, 135, 115], 'bicycle': [119, 11, 32]}

count = 0
for filename in glob.glob(args.input_dir + '/*'):
    print filename
    img_name, _ = os.path.splitext(os.path.basename(filename))
    dst_path = os.path.join(args.output_dir, img_name + ".png")

    image = Image.open(filename)

    npimage = np.array(image)
    npimage[np.where(
        np.logical_or(
            npimage == seg_class['obstacle'],
            npimage == seg_class['bicycle']).all(axis=2))] = [0, 0, 0]
    print npimage.shape
    count += 1
    if count == 5:
        break
    PILrgba = Image.fromarray(npimage)
    PILrgba.save(dst_path)
