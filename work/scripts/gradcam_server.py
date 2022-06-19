import os
import random
import flask
import torch
import numpy as np
from PIL import Image
import argparse
import json
from flask import Flask
from flask_cors import CORS
import math
from operator import itemgetter

from torch.utils.data import DataLoader


# create Flask app
app = Flask(__name__)
CORS(app)



@app.route('/image_concept_saliency', methods=['POST'])

def get_sample():

    client_data = flask.request.json
    
    img_index = client_data['img_index']

    score_metric = client_data['score_metric']

    score_ordering = client_data['ordering']

    if score_metric == 'iou':

    	data = iou_dict

    elif score_metric == 'gtc':

    	data = gtc_dict

    elif score_metric == 'sc':

    	data = sc_dict

    if score_ordering == 'ascending':

    	data_idx = data[img_index]

    elif score_ordering == 'descending':

    	data_idx = data[::-1][img_index]

    image_id = data_idx['image_id']
    concept_id = data_idx['concept_id']
    
    data = json.load(open(os.path.join(path, 'result_train', str(image_id)+'_'+str(concept_id)+'_'+str(percentile)+'_'+str(num_std)+'.json'),'r'))

    # result = {'image_id':image_filename,'concept_id':concept_id,'prediction':predictions,
    #     'num_components':num_components,'area_of_components':area_of_components,
    #     'saliency_method':saliency_method,'num_std': num_std,'percentile':percentile,
    #     'IoU':shared_interest_scores['iou_coverage'],'gtc':shared_interest_scores['ground_truth_coverage'], 
    #     'sc': shared_interest_scores['saliency_coverage']}


    return flask.jsonify(data)



@app.route('/result_img', methods=['POST'])

def get_result():
    
    data = result_list
    
    # data = json.load(open(os.path.join(path, 'result_train', str(image_id)+'_'+str(concept_id)+'_'+str(percentile)+'_'+str(num_std)+'.json'),'r'))

    # result = {'image_id':image_filename,'concept_id':concept_id,'prediction':predictions,
    #     'num_components':num_components,'area_of_components':area_of_components,
    #     'saliency_method':saliency_method,'num_std': num_std,'percentile':percentile,
    #     'IoU':shared_interest_scores['iou_coverage'],'gtc':shared_interest_scores['ground_truth_coverage'], 
    #     'sc': shared_interest_scores['saliency_coverage']}


    return flask.jsonify(data)


    
if __name__=='__main__':


    groundtruth_method = 'segmentation'

    saliency_method = 'GradCAM'

    saliency_mask_method = 'std'

    #path = './'+saliency_method+'/'+ground_truth_method+'/'+saliency_mask_method  #std or percentile
    #path = './output_imagenet_s/'
    path = './output_coco/'

    iou_dict = json.load(open(os.path.join(path, 'iou_list.json'),'r'))
    gtc_dict = json.load(open(os.path.join(path, 'gtc_list.json'),'r'))
    sc_dict = json.load(open(os.path.join(path, 'sc_list.json'),'r'))

    result_list = json.load(open(os.path.join(path, 'coco_result_brief.json'),'r'))

    percentile = 0.7
    num_std = 1
   
    app.run()
