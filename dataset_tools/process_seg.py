from PIL import Image
from pathlib import Path
import argparse
import numpy as np
import os
from tqdm import tqdm


def get_classes_to_turn_off(mode="translation"):
    classes_to_turn_off = {
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
    if mode == "translation":
        return classes_to_turn_off
    elif mode == "stuff_segmentation":
        classes_to_turn_off["fence"] = [190, 153, 153]
        classes_to_turn_off["tree"] = [51, 51, 0]
        classes_to_turn_off["obstacle"] = [2, 135, 115]
        return classes_to_turn_off


# Change grass color
GRASS_COLOR = [0, 102, 0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="path to folder containing segmented images")
    parser.add_argument("--output_dir", required=True, help="output path")
    parser.add_argument(
        "--mode",
        required=True,
        help="choose from [stuff_segmentation translation]")
    return parser.parse_args()


def process(input_dir, output_dir, seg_class):
    for filename in tqdm(os.listdir(input_dir)):
        filepath = Path(input_dir, filename)
        if os.path.isfile(filepath):

            dest_path = Path(output_dir, filename)
            image = Image.open(filepath)
            img_array = np.array(image)

            # Turn off classes
            img_array[np.where(
                np.logical_or.reduce(
                    (img_array == seg_class['door'],
                     img_array == seg_class['bicycle'],
                     img_array == seg_class['dirt'],
                     img_array == seg_class['gravel'],
                     img_array == seg_class['water'],
                     img_array == seg_class['fence-pole'],
                     img_array == seg_class['person'],
                     img_array == seg_class['dog'],
                     img_array == seg_class['bald-tree'],
                     img_array == seg_class['window'],
                     img_array == seg_class['air-marker'],
                     img_array == seg_class['conflicting'])).all(axis=2))] = [
                         0, 0, 0
                     ]

            # Change grass color
            img_array[np.where(
                (img_array == GRASS_COLOR).all(axis=2))] = [126, 126, 128]

            img_proc = Image.fromarray(img_array)
            img_proc.save(dest_path)


if __name__ == "__main__":
    args = get_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    seg_class = get_classes_to_turn_off(mode=args.mode)
    process(input_dir, output_dir, seg_class)