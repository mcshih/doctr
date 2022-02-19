'''
Make images transparent
'''
import cv2
import numpy as np
import os
from tqdm import tqdm

for root, dirnames, filenames in os.walk("/home/user/ACM/shih/IAM/words/"):
    pbar = tqdm(filenames)
    for filename in pbar:
        path = os.path.join(root, filename)
        try:
            img = cv2.imread(path)

            result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            for i in range(0,img.shape[0]):
                for j in range(0,img.shape[1]):
                    if img[i,j,0] > 200 and img[i,j,1] > 200 and img[i,j,2] > 200:
                        result[i,j,3] = 0
            
            root = root.replace("/words/", "/words_a/")
            pbar.set_description("Processing %s" % os.path.join(root, filename))
            cv2.imwrite(os.path.join(root, filename), result, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        except:
            print(path, "NOT SUCESS!")
        