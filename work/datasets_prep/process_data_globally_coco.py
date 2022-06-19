import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from tqdm import tqdm
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--annot-file', type=str, help='annotation file', default="./coco_dataset/annotations/instances_train2017.json")
parser.add_argument('--img-dir', type=str, help='image directory', default="./coco_dataset/train2017")
parser.add_argument('--output-dir', type=str, help='output directory', default="./output")
args = parser.parse_args()


# load coco data
coco = COCO(annotation_file=args.annot_file)

# get all image index info
img_ids = list(sorted(coco.imgs.keys()))
print(f"number of images: {len(img_ids)}")

# get all coco class labels
classes = dict([(v['id'], v['name']) for k, v in coco.cats.items()])

# iterate all images
for img_id in tqdm(img_ids):
    # get file name
    file_name = coco.loadImgs(img_id)[0]['file_name']
    if not os.path.isfile(os.path.join(args.img_dir, file_name)): continue

    img = Image.open(os.path.join(args.img_dir, file_name)).convert('RGB')
    img_w, img_h = img.size

    # get annotation info
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annots = coco.loadAnns(ann_ids)

    # generate json output
    results = []
    for ann in annots:
        x, y, w, h = list(map(int, ann['bbox']))

        ## bb_mask
        bb_mask = np.zeros((img_h, img_w), np.uint8)
        bb_mask = cv2.rectangle(bb_mask, (x, y), (x+w, y+h), (1, 1, 1), -1)

        ## seg_mask
        polygons = ann['segmentation']
        rles = coco_mask.frPyObjects(polygons, img_h, img_w)
        seg_mask = coco_mask.decode(rles)[..., 0]
        
        results.append({
            'image': file_name,
            'concept': ann['category_id'],
            'bb_mask': bb_mask.tolist(),
            'seg_mask': seg_mask.tolist()
        })
        
    # save as json
    save_path = Path(args.output_dir) / file_name
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    with open(str(save_path.with_suffix('.json')), 'w') as f:
        json.dump(results, f)
