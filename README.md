# Interaction 

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
* More details about ImageNet-S dataset could be read in https://github.com/UnsupervisedSemanticSegmentation/ImageNet-S

## 4. define path for importing package
```export PYTHONPATH=$PYTHONPATH:path/to/interaction```

## 3. Heatmap Dataset
* __CUB Dataset__
   * You could download the heatmaps of each filter pair and their intersection's activations on every image from ```zhaoy32/Desktop/Corr/corr_norm_activate/rank_img_activate_individual```
* __ImageNet Dataset__
   * For imagenet dataset, the heatmaps would be generated on the fly.

## 4. Server Dataset
* CUB Dataset
   * You could download the dataset for server from vandy box https://vanderbilt.box.com/s/87421wopi5xmb5lc8ndhsctnx8n97ei1

## 5. Initial Graph
* https://observablehq.com/d/d4bc687044fecaff
      
## 6. Steps 

* **CUB Dataset**
   * make a new file as local directory ```./```
   * Download the __image dataset__ from ```zhaoy32/Desktop/Corr/CUB_200_2011``` , save it to local directory, like ```./CUB_200_2011```
   * Download the **heatmap dataset** from ```zhaoy32/Desktop/Corr/corr_norm_activate/rank_img_activate_individual```, create necessary directories, to save the heatmap dataset to ```./corr_norm_activate/rank_img_activate_individual/```
   * Download **server dataset** from https://vanderbilt.box.com/s/87421wopi5xmb5lc8ndhsctnx8n97ei1 , save it to ```./```
   * Download **ordered_filter_pairs** from ```zhaoy32/Desktop/Corr/corr_norm_activate/ordered_filter_pairs.json```, save it to ```./```
   * run ```python server2.py``` under the local directory ```./```
   * run ```python -m http.server``` under the local directory ```./```
   * Open observable link https://observablehq.com/d/d4bc687044fecaff
   * the directory is similar like this:
   ```
   ./
   ├── CUB_200_2011
   │   └── CUB_200_2011
   │       └── images
   │           └── 001.Black_footed_Albatross
   ├── corr_norm_activate
   │   └── rank_img_activate_individual
   │       ├── rank_0
   │       ├── rank_1
   │       └── rank_img_list
   ├── graph_activate.json
   ├── image_filter_pairs.json
   ├── ordered_filter_pairs.json
   ├── ordered_filter_pairs_dict.json
   ├── thresholds.npy
   ├── self_iou.py
   └── server2.py

   ```
   
* **ImageNet Dataset**
   * Download the dictionary **imagenet_box** from ```zhaoy32/Desktop/Corr/imagenet_box/```
   * run ```python server2.py``` under the local directory ```./imagenet_box```
   * run ```python -m http.server``` under ```./imagenet_box/```
   * Open observable link https://observablehq.com/d/d4bc687044fecaff



## 7. Interactions
* Select the **threshold** for the __normalized intersection time__ for filter pairs on the slider
   * a new graph is shown based on the threshold.
* Click on an **edge** (filter pair 'src' and 'dst') in the graph
   * stroke of this edge will turn to **'pink'**.
   * a random set of 50 images where src and dst intersect with each other would be shown below, along with the heatmaps of the activation masks of src, dst, and their intersection, on each image.
   * To see more different images, you could turn the page by clicking on 'prev' and 'next' button.
   * The contour corresponding to the edge you clicked on will turn to 'green' color.
* Click on an **image** 
   * a **'yellow'** rectangle will highlight the image.
   * a 224*224 zoom-in version of that image would be shown.
   * all the filter pairs (edges) in the current graph, whose src and dst intersects on that image, will turn to 'blue'; 
   * In the meantime, the contours of their intersection masks would also be drawn on the zoom-in image, in **'pink'** .
* Click on an **unselected contour** (**'pink'**)
   * stroke of the contour would turn to **'red'**.
   * opacities of all the other contours would turn to 0.2.
   * The filter pair (edge) corresponding to that contour would turn to **'green'** in the graph.
* Click on a **selected contour** (**'red'**)
   * all of the contours in the zoom-in image would reverse to 'pink' color, and their opacity would all be 1 again.

    

