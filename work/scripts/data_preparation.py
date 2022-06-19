import json
import numpy as np
from PIL import Image


with open('/home/zhaoy32/Desktop/shared-interest/shared_interest/scripts/output_imagenet_s/result_imagenet_s_list.json') as f:
    result = json.load(f)


def flatten(json_result):
    keys = json_result[0].keys()
    res = {}
    for k in keys:
        data = np.array([d[k] for d in json_result])
        res[k] = data
    return res

result_dict = flatten(result)


with open('image_id.json', 'w') as f:
    json.dump(result_dict['image_id'].tolist(), f, indent=2)

result_dict['concept_id'].astype(np.int32).tofile('concept_id.bin')
result_dict['iou'].astype(np.float32).tofile('iou.bin')
result_dict['gtc'].astype(np.float32).tofile('gtc.bin')
result_dict['sc'].astype(np.float32).tofile('sc.bin')
