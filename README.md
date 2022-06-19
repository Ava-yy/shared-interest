# Interaction 

## 1.observable link
https://observablehq.com/d/2c8fc79ee08bb941
## 2. open server
```ssh -L 8000:localhost:8000 zhaoy32@10.2.191.150```
## 3. run server.py
* run server.py under ```zhaoy32/home/zhaoy32/Desktop/interaction/work```

# Prepare datasets from scratch
## 1. Clone this repo
## 2. Install Shared Interest 
Install the method locally for use in other development projects. It can be referenced as shared_interest within this package and in other locations.
```
cd shared-interest
pip install -e git+https://github.com/mitvis/shared-interest.git#egg=shared_interest
```
## 3. Download ImageNet dataset
* Download ILSVRC2012 validation dataset to ```./datasets/ILSVRC2012/val```  
* Or you could copy it from ```/home/zhaoy32/Desktop/shared-interest/datasets/ILSVRC2012```   
## 3. Create ImageNetS dataset
* clone ImageNet-S repo from https://github.com/UnsupervisedSemanticSegmentation/ImageNet-S to ```./datasets/```
* The ImageNet-S dataset is based on the ImageNet-1k dataset. You need to have a copy of ImageNet-1k dataset, and you can also get the rest of the ImageNet-S dataset (split/annotations) with the following command:
```
cd datapreparation
bash data_preparation.sh ./datasets/ILSVRC2012/val ./datasets/ILSVRC2012_Seg [split: 50 300 919 all] [whether to copy new images: false, true]
```
* Here we choose split:all, and use imagenetS919 validation dataset(12419 images) and semi-train datasets(9190 images). For each image, we generate the GradCAM heatmaps for the model's top 5 predictions. We also exclude some images with unwanted concepts.
* More details about ImageNet-S dataset could be read in https://github.com/UnsupervisedSemanticSegmentation/ImageNet-S

## 4. define path for importing package
```export PYTHONPATH=$PYTHONPATH:path/to/interaction```

## 5. calculate iou,gtc,sc scores 
* run ```run_prediction_top5.py``` under ```./work/scripts/```
* define the directories of outputs (change val to train if we use the 9k train-semi datasets); Here we use val first.

