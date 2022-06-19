import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from PIL import Image

from interpretability_methods import gradcam, vanilla_gradients
from shared_interest.datasets import get_dataset, get_transform
from shared_interest.shared_interest import shared_interest
from shared_interest.util import (binarize_percentile, binarize_std, component_analysis,
                    flatten, show_cam_on_image, save_grayscale)


# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='imagenet_s', choices=['imagenet_s', 'coco'])
parser.add_argument('--imagenets-dir', type=str, default="../datasets/ILSVRC2012_Seg/ImageNetS919/validation")
parser.add_argument('--imagenets-gt-dir', type=str, default="../datasets/ILSVRC2012_Seg/ImageNetS919/validation-segmentation")
#parser.add_argument('--imagenets-label-file', type=str, default="../datasets/ILSVRC2012_Seg/ImageNetS919/val_label_mapping.txt")
parser.add_argument('--coco-annot-file', type=str, default='../datasets/coco_dataset/annotations/instances_val2017.json')
parser.add_argument('--coco-image-dir', type=str, default='../datasets/coco_dataset/val2017')
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--saliency-method', type=str, default='std')
parser.add_argument('--ground-truth-method', type=str, default='seg_mask')
parser.add_argument('--percentile', type=float, default=0.5)
parser.add_argument('--num-std', type=float, default=1)
parser.add_argument('--output-dir', type=str, default='../outputs/output_imagenet_s')
args = parser.parse_args()

# load dataset
if args.dataset == 'coco':
    dataset = get_dataset(name='coco')(args.coco_annot_file, args.coco_image_dir, get_transform('image'), get_transform('ground_truth'))
elif args.dataset == 'imagenet_s':
    dataset = get_dataset(name='imagenet_s')(args.imagenets_dir, args.imagenets_gt_dir, get_transform('image'), get_transform('ground_truth'))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

model = torchvision.models.__dict__[args.arch](pretrained=True)
# load model
device = torch.device('cpu')
model = model.to(device)
# saliency method
saliency_method = gradcam.GradCAM(model, model.layer4[-1])
# saliency_method = vanilla_gradients.VanillaGradients(model)

unwanted_concepts = json.load(open('../datasets/unwanted_concepts.json','r'))

# create output folders
image_save_dir = os.path.join(args.output_dir,'image_crop_val')
mask_save_dir = os.path.join(args.output_dir ,'gt_mask_val')
saliency_heatmap_save_dir = os.path.join(args.output_dir ,'saliency_val' , 'saliency_heatmap')
saliency_mask_save_dir = os.path.join(args.output_dir , 'saliency_val' , 'saliency_mask')
result_save_dir = os.path.join(args.output_dir,'result_val')

if not image_save_dir.exists(): image_save_dir.mkdir(parents=True)
if not mask_save_dir.exists(): mask_save_dir.mkdir(parents=True)
if not saliency_heatmap_save_dir.exists(): saliency_heatmap_save_dir.mkdir(parents=True)
if not saliency_mask_save_dir.exists(): saliency_mask_save_dir.mkdir(parents=True)
if not result_save_dir.exists(): result_save_dir.mkdir(parents=True)

# eval mode
model.eval()

shared_interest_scores = {'iou_coverage': np.array([]),
                        'ground_truth_coverage': np.array([]),
                        'saliency_coverage': np.array([]),}

result_list = []

iou_list = []
gtc_list = []
sc_list = []

# load imagenetS919 label id -> original imagenet label id:
imagenet_s_label_dict = json.load(open('../datasets/imagenetS_label_dict.json','r'))

# main loop
for i, (image_names, images, bb_masks, seg_masks, labels) in enumerate(tqdm(dataloader)):

    print('image_name, labels : ',image_names,labels)
    # save images
    for j in range(len(image_names)):
        image_name = image_names[j]
        image = images[j]
        bb_mask = bb_masks[j]
        seg_mask = seg_masks[j]
        label = labels[j] # tensor

        pimage: Image.Image = get_transform('reverse_image')(image)
        pimage.save(image_save_dir / (image_name + '_crop.jpg'))
        # cv2.imwrite(str(mask_save_dir / (image_name + f'_{label}' + '_bbox_crop.jpg')), bb_mask.numpy() * 255)
        cv2.imwrite(str(mask_save_dir / (image_name + f'_{label}' + '_segm_crop.jpg')), seg_mask.numpy() * 255)


    with torch.no_grad():
        images = images.to(device)
        if args.ground_truth_method == 'seg_mask':
            ground_truth = seg_masks.numpy()
        else:
            ground_truth = bb_masks.numpy()
        labels = labels.numpy()

        output = model(images)
        predictions = torch.argmax(output, dim=1).item() # model output is from 0 - 999; consistent with the original ImageNet dataset (0-999 labels);
        top5_predictions = torch.argsort(output,dim=1,descending=True)
        # print('top5_predictions : ',top5_predictions[0][:5])
        top5_predictions = top5_predictions[0][:5] # torch.tensor

    num_components_gt, area_of_components_gt = component_analysis(ground_truth)
    
    # must set batch_size as 1
    assert len(image_names) == 1

    # print(type(labels),labels,type(label),label)  # <class 'numpy.ndarray'> [1] <class 'torch.Tensor'> tensor(1, dtype=torch.int32)
    # print(torch.cat((top5_predictions,label.unsqueeze(0))))

    label = int(imagenet_s_label_dict[str(labels[0])])  # convert label id from ImageNetS (1-919) to label id in ImageNet (0-999)    label from ImageNetS ranging from 1-919, is different from the original ImageNet label from 0 - 999;
    label = torch.tensor(label)


    for concept_id in torch.cat((top5_predictions,label.unsqueeze(0))):

        if concept_id.item() in unwanted_concepts:
            continue

        gradcam, upsampled_gradcam = saliency_method.get_saliency(image.unsqueeze(0), target_classes=torch.tensor(concept_id).unsqueeze(0))
        save_grayscale(str(saliency_heatmap_save_dir / (image_name + f'_{concept_id}' + '_gradcam.jpg')), gradcam.detach().numpy())
        save_grayscale(str(saliency_heatmap_save_dir / (image_name + f'_{concept_id}' + '_upsample_gradcam.jpg')), upsampled_gradcam)

        # save gradcam as images
        rgb_img = np.array(pimage)
        upsampled_gradcam = (upsampled_gradcam - upsampled_gradcam.min()) / (upsampled_gradcam.max() - upsampled_gradcam.min())
        cam_img = show_cam_on_image(rgb_img / 255, upsampled_gradcam.squeeze(0).squeeze(0), use_rgb=True)
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(saliency_heatmap_save_dir / (image_name + f'_{concept_id}' + '_heatmap.jpg')), cam_img)

        # save segmentation result
        seg_img = show_cam_on_image(rgb_img / 255, seg_mask.numpy().astype(np.float32), use_rgb=True)
        cv2.imwrite(str(mask_save_dir / (image_name + f'_{concept_id}' + '_segmentation.jpg')), seg_img)

        # save saliency masks
        saliency = flatten(upsampled_gradcam)
        if args.saliency_method == 'percentile':
            saliency_masks = binarize_percentile(saliency, percentile=args.percentile)
        elif args.saliency_method == 'std':
            saliency_masks = binarize_std(saliency, num_std=args.num_std)
        cv2.imwrite(str(saliency_mask_save_dir / (image_name + f'_{concept_id}' + '_saliencymask' + f'_{args.percentile}_{args.num_std}.jpg')), saliency_masks[0] * 255)

        num_components_sc, area_of_components_sc = component_analysis(saliency_masks)

        for score in shared_interest_scores:
            shared_interest_scores[score] = shared_interest(ground_truth, saliency_masks, score=score)

        result_img = dict(
            image_id = image_names[0],
            concept_id = concept_id.item(),
            label_id = int(labels[0]),
            top5_prediction = top5_predictions.tolist(),
            num_components_sc=num_components_sc,
            area_of_components_sc = area_of_components_sc,
            num_components_gt = num_components_gt,
            area_of_components_gt = area_of_components_gt,
            saliency_method = args.saliency_method,
           # percentile = float(args.percentile),
           # num_std = float(args.num_std),
            iou = float(shared_interest_scores['iou_coverage'][0]),
            gtc = float(shared_interest_scores['ground_truth_coverage'][0]),
            sc = float(shared_interest_scores['saliency_coverage'][0])
        )

        result_list.append(result_img)

        iou_list.append(dict(image_id=image_name, concept_id=concept_id.item(), score=shared_interest_scores['iou_coverage'][0]))
        gtc_list.append(dict(image_id=image_name, concept_id=concept_id.item(), score=shared_interest_scores['ground_truth_coverage'][0]))
        sc_list.append(dict(image_id=image_name, concept_id=concept_id.item(), score=shared_interest_scores['saliency_coverage'][0]))

iou_list = sorted(iou_list, key=lambda d: d['score']) 
gtc_list = sorted(gtc_list, key=lambda d: d['score']) 
sc_list = sorted(sc_list, key=lambda d: d['score'])


json.dump(result_list, open(result_save_dir / ('result_imagenet_s_list_'+ f'_{args.percentile}_{args.num_std}.json'), 'w'), indent=4)

# dump scores
json.dump(iou_list, open(str(Path(args.output_dir) / 'iou_list.json'), 'w'))
json.dump(gtc_list, open(str(Path(args.output_dir) / 'gtc_list.json'), 'w'))
json.dump(sc_list, open(str(Path(args.output_dir) / 'sc_list.json'), 'w'))


def flatten(json_result):
    keys = json_result[0].keys()
    res = {}
    for k in keys:
        data = np.array([d[k] for d in json_result])
        res[k] = data
    return res

result_dict = flatten(result_list)


with open('image_id.json', 'w') as f:
    json.dump(result_dict['image_id'].tolist(), f, indent=2)

result_dict['concept_id'].astype(np.int32).tofile('concept_id.bin')
result_dict['iou'].astype(np.float32).tofile('iou.bin')
result_dict['gtc'].astype(np.float32).tofile('gtc.bin')
result_dict['sc'].astype(np.float32).tofile('sc.bin')
result_dict['num_components_sc'].astype(np.float32).tofile('num_components_sc.bin')
result_dict['num_components_gt'].astype(np.float32).tofile('num_components_gt.bin')

area_comp_gt = [np.array(item).sum() for item in result_dict['area_of_components_gt']]
area_comp_sc = [np.array(item).sum() for item in result_dict['area_of_components_sc']]
area_comp_sc = np.array(area_comp_sc)
area_comp_gt = np.array(area_comp_gt)
area_comp_gt.astype(np.float32).tofile('area_of_components_gt.bin')
area_comp_sc.astype(np.float32).tofile('area_of_components_sc.bin')







