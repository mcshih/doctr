import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from doctr.models.predictor import OCRPredictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models import crnn_vgg16_bn, db_resnet50
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page

# Instantiate your model here
det_model = db_resnet50(pretrained=False)
reco_model = crnn_vgg16_bn(pretrained=True)
det_params = torch.load("/home/user/ACM/shih/doctr/IMGUR5K_shrink.pt", map_location="cpu")
det_model.load_state_dict(det_params)

det_predictor = DetectionPredictor(PreProcessor((1024, 1024), batch_size=1), det_model)
reco_predictor = RecognitionPredictor(PreProcessor((32, 128), preserve_aspect_ratio=True, batch_size=32), reco_model)

predictor = OCRPredictor(det_predictor, reco_predictor)
predictor.cuda(0)

imgs_folder = "/home/user/ACM/shih/FUNSD/dataset/testing_data/images/"
save_folder = "/home/user/ACM/shih/FUNSD/demo/"

files = os.listdir(imgs_folder)
pbar = tqdm(files)
for file in pbar:
    pbar.set_description("Processing %s" % file)
    img = DocumentFile.from_images(imgs_folder + file)
    result = predictor(img)
    #print(type(img))
    output = visualize_page(result.pages[0].export(), np.asarray(img[0]))
    output.savefig(save_folder + file)
    plt.close(output)
