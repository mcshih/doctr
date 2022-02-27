from operator import gt
import cv2
import numpy as np
import os
import torch

os.environ['USE_TORCH'] = '1'

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
from doctr.models.predictor import OCRPredictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models import crnn_vgg16_bn, db_resnet50, ocr_predictor
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
from doctr.utils.metrics import LocalizationConfusion, OCRMetric, TextMatch

device = torch.device("cuda:0")
torch.cuda.set_device(0)

def pred_boxes_list(result):
    pred_boxes = []
    height, width = result.pages[0].dimensions
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (a, b), (c, d) = word.geometry
                pred_boxes.append([int(a * width), int(b * height), int(c * width), int(d * height)])
    return pred_boxes

# Instantiate your model here
det_model = db_resnet50(pretrained=False)
reco_model = crnn_vgg16_bn(pretrained=True)
det_params = torch.load("/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/doctr/baseline_mergedataset_3.pt", map_location='cpu')
det_model.load_state_dict(det_params)

det_predictor = DetectionPredictor(PreProcessor((1024, 1024), batch_size=1), det_model)
reco_predictor = RecognitionPredictor(PreProcessor((32, 128), preserve_aspect_ratio=True, batch_size=32), reco_model)

predictor = OCRPredictor(det_predictor, reco_predictor)

predictor = predictor.eval().to(device=device)

imgs_folder = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/FUNSD/dataset/testing_data/images/"
save_folder = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/FUNSD/demo(myown_3)/"

files = os.listdir(imgs_folder)
pbar = tqdm(files)
for idx, file in enumerate(pbar):
    '''
    if idx > 0:
        break
    '''
    pbar.set_description("Processing %s" % file)
    img = DocumentFile.from_images(os.path.join(imgs_folder, file))
    orgin_img = cv2.imread(os.path.join(imgs_folder, file))
    result = predictor(img)
    pred_boxes = pred_boxes_list(result)
    
    # save file
    for coord in pred_boxes:
        xmin,ymin,xmax,ymax = coord
        orgin_img = cv2.rectangle(orgin_img, (xmin,ymin), (xmax,ymax), (255,0,0), 2)
    cv2.imwrite(os.path.join(save_folder, file), orgin_img)
    