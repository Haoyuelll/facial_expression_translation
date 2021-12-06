import cv2
import os
import json

for file in os.listdir('./image'):
    image = cv2.imread(file, -1)
    mask = image.copy()
    num_channels = 1 if len(mask.shape) == 2 else mask.shape[2]

    bbox_dict = json.load('./Bbox.json')
    features = ["eyebrow1", "eyebrow2", "nose", "nosetril", "eye1", "eye2", "lips", "teeth"]
    for feature in features:
        xmin, ymin, xmax, ymax = bbox_dict[file[:-4]][feature]
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color=(0, 0, 0) * num_channels)

    cv2.imwrite(f'./mask/{file[:-4]}_masked.{file[-4:]}', mask)
