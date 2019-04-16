from PIL import Image
from pathlib import Path
import argparse
import numpy as np
import os
from tqdm import tqdm

# Classes to turn off
SEG_CLASS = {
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
SEG_CLASS['grass'] = [0, 102, 0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="path to folder containing segmented images")
    parser.add_argument("--output_dir", required=True, help="output path")
    return parser.parse_args()


def process(input_dir, output_dir):
    for filename in tqdm(os.listdir(input_dir)):
        filepath = Path(input_dir, filename)
        if os.path.isfile(filepath):

            dest_path = Path(args.output_dir, filename)
            image = Image.open(filepath)
            img_array = np.array(image)

            # Turn off classes
            img_array[np.where(
                np.logical_or.reduce(
                    (img_array == SEG_CLASS['door'],
                     img_array == SEG_CLASS['bicycle'],
                     img_array == SEG_CLASS['dirt'],
                     img_array == SEG_CLASS['gravel'],
                     img_array == SEG_CLASS['water'],
                     img_array == SEG_CLASS['fence-pole'],
                     img_array == SEG_CLASS['person'],
                     img_array == SEG_CLASS['dog'],
                     img_array == SEG_CLASS['bald-tree'],
                     img_array == SEG_CLASS['window'],
                     img_array == SEG_CLASS['air-marker'],
                     img_array == SEG_CLASS['conflicting'])).all(axis=2))] = [
                         0, 0, 0
                     ]

            # Change grass color
            img_array[np.where(
                (img_array == SEG_CLASS['grass']).all(axis=2))] = [
                    126, 126, 128
                ]

            img_proc = Image.fromarray(img_array)
            img_proc.save(dest_path)


if __name__ == "__main__":
    args = get_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    process(input_dir, output_dir)