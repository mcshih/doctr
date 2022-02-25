from operator import gt
import cv2
import numpy as np
import os

os.environ['USE_TORCH'] = '1'

import torch
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

def _pct(val):
    return "N/A" if val is None else f"{val:.2%}"

def xml_parser(xmL_file):
    tree = ET.parse(xmL_file)
    root = tree.getroot()
    total_height, total_width = int(root.attrib['height']), int(root.attrib['width'])
    
    gt_boxes = []
    for word in root.iter('word'):
        x_min, x_max = total_width, 0
        y_min, y_max = total_height, 0
        for cmp in word:
            x_left, x_right = int(cmp.attrib['x']), int(cmp.attrib['x'])+int(cmp.attrib['width'])
            y_up, y_down = int(cmp.attrib['y']), int(cmp.attrib['y'])+int(cmp.attrib['height'])
            x_min = min(x_left, x_min)
            x_max = max(x_right, x_max)
            y_min = min(y_up, y_min)
            y_max = max(y_down, y_max)
        #print("{} bbox:[ {}, {}, {}, {}]".format(word.attrib['text'],x_min, y_min, x_max, y_max))
        gt_boxes.append([x_min, y_min, x_max, y_max])
    return gt_boxes

device = torch.device("cuda")
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
det_params = torch.load("/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/doctr/baseline_mergedataset_2.pt", map_location='cpu')
det_model.load_state_dict(det_params)

det_predictor = DetectionPredictor(PreProcessor((1024, 1024), batch_size=1), det_model)
reco_predictor = RecognitionPredictor(PreProcessor((32, 128), preserve_aspect_ratio=True, batch_size=32), reco_model)

predictor = OCRPredictor(det_predictor, reco_predictor)

pretrained_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
# predictor = predictor.cuda()
det_metric = LocalizationConfusion(iou_thresh=0.5)
det_pretrain_metric = LocalizationConfusion(iou_thresh=0.5)

imgs_folder = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/IAM/forms/"
xml_folder = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/IAM/xml/"
save_folder = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/demo(IMGUR5K_shrink)/"

files = os.listdir(imgs_folder)
pbar = tqdm(files)
for idx, file in enumerate(pbar):
    '''
    if idx > 0:
        break
    '''
    pbar.set_description("Processing %s" % file)
    xml_file = file.replace('.png','.xml')
    img = DocumentFile.from_images(os.path.join(imgs_folder, file))
    result = predictor(img)
    pretrained_result = pretrained_model(img)
    pred_boxes = pred_boxes_list(result)
    pred_pretrained_boxes = pred_boxes_list(pretrained_result)
    #print(pred_boxes)
    gt_boxes = xml_parser(os.path.join(xml_folder+xml_file))
    #print(gt_boxes)
    det_metric.update(np.asarray(gt_boxes), np.asarray(pred_boxes))
    det_pretrain_metric.update(np.asarray(gt_boxes), np.asarray(pred_pretrained_boxes))
    # save file
    '''
    output = visualize_page(result.pages[0].export(), np.asarray(img[0]))
    output.savefig(save_folder + file)
    plt.close(output)
    '''
recall, precision, mean_iou = det_metric.summary()
print(f"Text Detection - Recall: {_pct(recall)}, Precision: {_pct(precision)}, Mean IoU: {_pct(mean_iou)}")
recall, precision, mean_iou = det_pretrain_metric.summary()
print(f"Text Detection - Recall: {_pct(recall)}, Precision: {_pct(precision)}, Mean IoU: {_pct(mean_iou)}")