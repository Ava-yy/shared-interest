from .imagenet_s_dataset import ImageNetSDataset

import torchvision.transforms as transforms
from PIL import Image


def get_transform(name='image'):
    if name == 'image':
        return transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    elif name == 'ground_truth':
        return transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256, Image.NEAREST),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    elif name == 'reverse_image':
        return transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
                                    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
                                    transforms.ToPILImage(),])
    else:
        raise KeyError("transform name is invalid!")

def get_dataset(name='imagenet'):
    if name == 'imagenet_s':
        return ImageNetSDataset
#     elif name == 'imagenet':
#         return ImageNetDataset
#     elif name == 'coco':
#         return CocoDataset
    else:
        raise KeyError("dataset name is invalid!")
