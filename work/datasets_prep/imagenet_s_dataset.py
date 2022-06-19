"""Dataset for ImageNet with annotations."""

import os
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

import torch
from torchvision.datasets import ImageFolder


class ImageNetSDataset(ImageFolder):
    """Extends ImageFolder dataset to include ground truth annotations."""

    def __init__(self, image_path, ground_truth_path, image_transform=None, ground_truth_transform=None):
        """
        Extends the parent class with annotation information.

        Additional Args:
        image_path: the path to the ImageNet images. This folder must be
            formatted in ImageFolder style (i.e. label/imagename.jpeg)
        ground_truth_path: the path to the ImageNet annotations. This folder
            must be formated in ImageFolder style (i.e., label/imagename.xml).
        image_transform: a pytorch transform to apply to the images or None.
            Defaults to None.
        ground_truth_transform: a pytorch transform to apply to the ground
            truth annotations or None. Defaults to None.

        """
        super().__init__(image_path, transform=image_transform)

        self.image_path = image_path
        self.ground_truth_path = ground_truth_path

        self.ground_truth_transform = ground_truth_transform
        self.ground_truth_path = ground_truth_path


    def __getitem__(self, index):
        """Returns the image, ground_truth mask, and label of the image."""
        image, _ = super().__getitem__(index)
        img_path, _ = self.imgs[index]
        image_name = img_path.strip().split('/')[-1].split('.')[0]

        # generate masks
        ground_truth_file = img_path.replace('validation', 'validation-segmentation').replace('.JPEG', '.png')
        bb_mask, seg_mask, label = self._create_masks(ground_truth_file)

        if self.ground_truth_transform != None:
            bb_mask = self.ground_truth_transform(bb_mask).squeeze(0)
            seg_mask = self.ground_truth_transform(seg_mask).squeeze(0)

        # get label
        # label = int(self.img2label[image_name]) - 1

        return image_name, image, bb_mask, seg_mask, label

    def _create_masks(self, ground_truth_file):
        """Creates a binary groudn truth mask based on the ImageNet annotations."""
        segmentation = Image.open(ground_truth_file).convert('RGB')
        segmentation = np.array(segmentation).astype(np.int32)
        segmentation_id = segmentation[:, :, 0] + segmentation[:, :, 1] * 256           # R+G*256
        non_ignored = ~(segmentation_id == 1000) # The ignored part is annotated as 1000, and the other category is annotated as 0.
        segmentation_id = segmentation_id * non_ignored
        class_id = segmentation_id.max()
        segmentation_mask = np.divide(segmentation_id, class_id).astype(np.uint8)

        # # generate bb mask
        # contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        height, width = segmentation_mask.shape
        bb_mask = torch.zeros((height, width), dtype=torch.int32)
        # for contour in contours:
        #     x_min, y_min = contour.min(axis=0).flatten()
        #     x_max, y_max = contour.max(axis=0).flatten()
        #     bb_mask[y_min:y_max, x_min:x_max] = 1

        # generate seg mask
        seg_mask = torch.tensor(segmentation_mask).to(torch.int32)

        return bb_mask, seg_mask, class_id
