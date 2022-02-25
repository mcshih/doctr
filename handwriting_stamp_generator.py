import cv2
import numpy as np
import os
import json
import random
import hashlib
from tqdm import tqdm

# Word List
word_file_list = []
for root, dirnames, filenames in os.walk("/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/IAM/words_a/"):
    for filename in filenames:
        path = os.path.join(root, filename)
        word_file_list.append(path)

hk_path = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/HK_dataset/img_a"
files = os.listdir(hk_path)
for file in files:
    path = os.path.join(hk_path, file)
    word_file_list.append(path)

def images_process(image_path, image_final_name):
    result_path = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/DDI-100/my_train_dataset/"

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    
    height, width = img.shape[:2]
    N_word = random.randint(1, 20)
    N_selected_words = random.sample(word_file_list, N_word)

    polys = []
    for idx, word_path in enumerate(N_selected_words):
        try:
            word_img = cv2.imread(word_path, cv2.IMREAD_UNCHANGED)
            word_height, word_width = word_img.shape[:2]
            
            height_ = random.randint(word_height+5, height-word_height-5)
            width_ = random.randint(word_width+5, width-word_width-5)
        except:
            print("ERROR:",image_path, word_path)
            continue

        alpha_s = word_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        # Rect: x, y, width, height
        Rect = [(width_, height_), (word_width, word_height), 0]
        # print('#{}(shape:{}): {}'.format(idx, word_img.shape,Rect))
        rectCnt = np.int_(cv2.boxPoints(Rect))
        polys.append(cv2.boxPoints(Rect).tolist())
        #print(rectCnt)
        
        # draw bbox
        # cv2.drawContours(canvas, [rectCnt], 0, (0,255,0), 3)

        y1= min(rectCnt[:,1])
        x1= min(rectCnt[:,0])
        for c in range(0, 3):
            canvas[y1:y1+word_height, x1:x1+word_width, c] = (alpha_s * word_img[:, :, c] +alpha_l * canvas[y1:y1+word_height, x1:x1+word_width, c])
    cv2.imwrite(os.path.join(result_path, image_final_name), canvas)

    # perform annotations
    img_dict = {}
    img_dict['img_dimensions'] = (height, width)
    with open(os.path.join(result_path, image_final_name), "rb") as f:
        img_dict['img_hash'] = hashlib.sha256(f.read()).hexdigest()
    img_dict['polygons'] = polys
    return img_dict


# main

#images_process("/home/user/ACM/shih/DDI-100/dataset_v1.3/01/orig_texts/0.png")

label_file = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/DDI-100/04_gen_my_labels.json"
labels_dict = {}
origin_img_folder = "/mnt/baf69772-7c2f-4570-a192-06c62f849660/data/shih/DDI-100/dataset_v1.3/04/gen_imgs/"
pbar = tqdm(os.listdir(origin_img_folder))
for doc in pbar:
    #print(doc)
    save_name = "04_"+doc
    labels_dict[save_name] =  images_process(os.path.join(origin_img_folder,doc), save_name)
with open(label_file, "w") as outfile:
    json.dump(labels_dict, outfile, indent = 4)
    outfile.close()

