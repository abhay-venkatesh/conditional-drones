from PIL import Image
import argparse
import glob
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir",
    required=True,
    help="path to folder containing segmented images")
parser.add_argument("--output_dir", required=True, help="output path")
args = parser.parse_args()

# Classes to turn off
seg_class = {
    'bicycle': [119, 11, 32],
    'dirt': [130, 76, 0],
    'gravel': [112, 103, 87],
    'water': [28, 42, 168],
    'fence-pole': [153, 153, 153],
    'person': [255, 22, 96],
    'dog': [102, 51, 0],
    'bald-tree': [190, 250, 190],
    'air-marker': [112, 150, 146],
    'conflicting': [255, 0, 0],
    'door': [254, 148, 12],
    'window': [254, 228, 12]
}

# Change grass color
seg_class['grass'] = [0, 102, 0]

count = 0
for filename in glob.glob(args.input_dir + '/*'):
    img_name, _ = os.path.splitext(os.path.basename(filename))
    dst_path = os.path.join(args.output_dir, img_name + ".png")
    image = Image.open(filename)
    npimage = np.array(image)
    
    # Turn off classes
    npimage[np.where(
        np.logical_or.reduce(
            (npimage == seg_class['door'], npimage == seg_class['bicycle'],
             npimage == seg_class['dirt'], npimage == seg_class['gravel'],
             npimage == seg_class['water'], npimage == seg_class['fence-pole'],
             npimage == seg_class['person'], npimage == seg_class['dog'],
             npimage == seg_class['bald-tree'], npimage == seg_class['window'],
             npimage == seg_class['air-marker'],
             npimage == seg_class['conflicting'])).all(axis=2))] = [0, 0, 0]

    # Change grass color
    npimage[np.where(
        (npimage == seg_class['grass']).all(axis=2))] = [126, 126, 128]
    count += 1
    if count % 10 == 0:
        print('Number of images processed : {}'.format(count))
    PILrgba = Image.fromarray(npimage)
    PILrgba.save(dst_path)
