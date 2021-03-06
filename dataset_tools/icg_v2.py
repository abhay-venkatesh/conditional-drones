"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
	   the command line as such:

	# Train a new model starting from pre-trained COCO weights
	python3 icg.py train --dataset=/path/to/icg/dataset --weights=coco

	# Resume training a model that you had trained earlier
	python3 icg.py train --dataset=/path/to/icg/dataset --weights=last

	# Train a new model starting from ImageNet weights
	python3 icg.py train --dataset=/path/to/icg/dataset --weights=imagenet

	# evaluate using specific weights
	python3 icg.py val --weights=/path/to/weights/file.h5 --image=<URL or path to file>

	# evaluate using specific weights
	python3 icg.py val --weights=last --image=<URL or path to file>
"""

import os
import sys
import json
import datetime
import random
import numpy as np
import skimage.draw
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage import filters

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.model import log
from mrcnn import visualize

# Handling NSInvalidArgumentException on OSX
from sys import platform as sys_pf
print(sys_pf)
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

#RGB_classes = {(128, 64, 128): 'paved-area',
#			   (48, 41, 30): 'rocks',
#			   (0, 50, 89): 'pool',
#			   (28, 42, 168): 'water',
#			   (107, 142, 35): 'vegetation',
#			   (70, 70, 70): 'roof',
#			   (102, 102, 156): 'wall',
#			   (190, 153, 153): 'fence',
#			   (9, 143, 150): 'car',
#			   (51, 51, 0): 'tree',
#			   (2, 135, 115): 'obstacle',
#			   (0, 102, 0): 'grass' }
RGB_classes = {
    (9, 143, 150): 'car',
    (48, 41, 30): 'rocks',
    (70, 70, 70): 'roof',
    (190, 153, 153): 'fence',
    (51, 51, 0): 'tree'
}

############################################################
#  Configurations
############################################################


class IcgConfig(Config):
    NAME = "icg"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(RGB_classes)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.0


############################################################
#  Dataset
############################################################


class IcgDataset(utils.Dataset):
    def load_icg(self, dataset_dir, subset):

        assert subset in ["train", "val"]
        images_dir = os.path.join(dataset_dir, 'images', subset)

        files = []
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            files.extend(filenames)

        # Add classes
        index = 0
        for class_val in RGB_classes.values():
            self.add_class("icg", index, class_val)
            index += 1

        # Add images
        for f in files:
            img_id = str(f[:-4])
            self.add_image(
                "icg",
                image_id=img_id,
                path=os.path.join(images_dir, img_id + '.jpg'),
                mask_image_path=os.path.join(dataset_dir, 'images_seg',
                                             img_id + '.png'))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
	   Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
        # If not a icg dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "icg":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask_img_path = info['mask_image_path']
        color_image = imread(mask_img_path)

        label_img = label(color_image)

        # each connected-component will have its own label
        # it will repeat 3 times for RGB so only use every 3rd
        seen = {}
        height, width = color_image.shape[:2]
        instance = 0
        instance_masks = []
        class_ids = []
        regions = regionprops(label_img)

        region_idx = []
        for idx, region in enumerate(regions):
            bbox = region.bbox
            if (bbox[1], bbox[0], bbox[4], bbox[3]) in seen:
                continue  # we've seen this object

            region_idx.append(idx)
            seen[(bbox[1], bbox[0], bbox[4], bbox[3])] = True

        mask = np.zeros([height, width, len(region_idx)], dtype=np.uint8)

        for idx in region_idx:
            region = regions[idx]

            # find centroid coords, only need row and col
            centroid = np.rint(region.centroid[0:2]).astype(np.int32)
            # get corresponding RGB color and label
            RGB_key = tuple(color_image[centroid[0], centroid[1]])

            if RGB_key not in RGB_classes:
                class_name = 'BG'  # we don't care about this label
            else:
                class_name = RGB_classes[RGB_key]
            for ele in self.class_info:
                if ele['name'] == class_name:
                    class_ids.append(ele['id'])
                    break

            mask[region.coords[:, 0], region.coords[:, 1], instance] = 1
            instance += 1

        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "icg":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = IcgDataset()
    dataset_train.load_icg(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = IcgDataset()
    dataset_val.load_icg(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=30,
        layers='heads')


############################################################
#  Evaluate
############################################################


def evaluate(model, inference_config):
    dataset_val = IcgDataset()
    dataset_val.load_icg(args.dataset, "val")
    dataset_val.prepare()

    image_id = random.choice(dataset_val.image_ids)

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        dataset_val, inference_config, image_id, use_mini_mask=False)
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    #visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_val.class_names, show_bbox=False, show_mask=False, title="Predictions")
    results = model.detect([original_image], verbose=1)

    r = results[0]
    #visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], ax=get_ax())

    image_ids = np.random.choice(dataset_val.image_ids, 40)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
         modellib.load_image_gt(dataset_val, inference_config,
                 image_id, use_mini_mask=False)
        molded_images = np.expand_dims(
            modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
         utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
              r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects in icg dataset.')
    parser.add_argument(
        "command", metavar="<command>", help="'train' or 'splash'")
    parser.add_argument(
        '--dataset',
        required=False,
        metavar="/path/to/icg/dataset/",
        help='Directory of the Icg dataset')
    parser.add_argument(
        '--weights',
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'")
    parser.add_argument(
        '--logs',
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = IcgConfig()
    else:

        class InferenceConfig(IcgConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(
            mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(
            mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
                "mrcnn_mask"
            ])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "val":
        evaluate(model, config)
