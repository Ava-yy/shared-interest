import os
import torch

from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):

    def __init__(self, annot_file, image_dir, image_transfrom=None, ground_truth_transform=None) -> None:
        super().__init__()

        self.image_dir = image_dir
        self.image_transform = image_transfrom
        self.ground_truth_transform = ground_truth_transform

        self.coco = COCO(annotation_file=annot_file)

        # get image ids
        img_ids = list(sorted(self.coco.imgs.keys()))

        # gather concept info
        self.data = []
        for img_id in img_ids:
            # get file name
            file_name = self.coco.loadImgs(img_id)[0]['file_name']

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annots = self.coco.loadAnns(ann_ids)

            concept_dic = {}
            for ann in annots:
                concept_id = ann['category_id']
                if concept_id not in concept_dic:
                    concept_dic[concept_id] = dict(bboxs=[], segmentations=[])
                concept_dic[concept_id]['bboxs'].append(ann['bbox'])
                concept_dic[concept_id]['segmentations'].append(ann['segmentation'])
            
            for concept in concept_dic.keys():
                self.data.append(
                    dict(
                        file_name = file_name,
                        concept = concept,
                        bboxs = concept_dic[concept]['bboxs'],
                        segmentations = concept_dic[concept]['segmentations']
                    )
                )

    def __getitem__(self, index):
        item = self.data[index]
        image_name = item['file_name'].split('.')[0]
        image_path = os.path.join(self.image_dir, item['file_name'])
        label = item['concept']

        raw_image = Image.open(image_path).convert('RGB')
        width, height = raw_image.size
        image = self.image_transform(image)
    
        bb_mask = self._create_bb_mask(item['bbox'], width, height)
        seg_mask = self._create_seg_mask(item['segmentation'], width, height)
        if self.ground_truth_transform != None:
            bb_mask = self.ground_truth_transform(bb_mask).squeeze(0)
            seg_mask = self.ground_truth_transform(seg_mask).squeeze(0)
        return image_name, image, bb_mask, seg_mask, int(label)

    def _create_bb_mask(self, bboxs, width, height):
        mask = torch.zeros((height, width))
        for bbox in bboxs:
            x, y, w, h = list(map(int, bbox))
            mask[y:y+h, x:x+w] = 1
        return mask
    
    def _create_seg_mask(self, segmentations, width, height):
        mask = torch.zeros((height, width))
        for segm in segmentations:
            rles = coco_mask.frPyObjects(segm, height, width)
            mask += coco_mask.decode(rles)[..., 0]
        return (mask > 0).to(int)

    def __len__(self):
        return len(self.data)
