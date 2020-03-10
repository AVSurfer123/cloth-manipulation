import sys
sys.path.insert(1, '/home/owen/anaconda2/envs/softlearning/lib/python3.6/site-packages/cv2/')
import cv2
import numpy as np
import os
import glob
from collections import defaultdict
from pprint import pprint
import matplotlib.pyplot as plt

from vision_utils import segment_image, GREEN, WHITE, BLUE, YELLOW, IMAGE_INPUT_SIZE


IMAGE_DIR = "/home/owen/wilson/cloth-manipulation/images/"
EXP_PATH = os.path.join(IMAGE_DIR, "rope_test")

GOAL_IMAGE_PATHS = [os.path.join(IMAGE_DIR, file) for file in ["rope_goal_flat.png", "vert_rope.png", "pi_over_4.png", "3pi_over_4.png", "squiggle.png"]]
SEGMENTED_GOAL_IMAGES = [segment_image(cv2.resize(cv2.imread(image_name), (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)), WHITE) for image_name in GOAL_IMAGE_PATHS]
GOAL_DICT = dict(zip(["horizontal", "vertical", "_pi_over_4", "3pi_over_4", "squiggle"], SEGMENTED_GOAL_IMAGES))

for img in GOAL_DICT:
    plt.imshow(GOAL_DICT[img])
    plt.show()

EXP_RESULTS = defaultdict(list)

for folder in sorted(os.listdir(EXP_PATH)):
    folder_path = os.path.join(EXP_PATH, folder)
    # print("Folder", folder)
    if "rope" in folder:
        for exp in sorted(os.listdir(folder_path)):
            exp_path = os.path.join(folder_path, exp)
            print("Exp", exp)
            for goal_name in GOAL_DICT.keys():
                if goal_name in exp:
                    best_img = None
                    max_intersection = float("-inf")
                    for image in os.listdir(os.path.join(exp_path, "segmentations")):
                        image_path = os.path.join(exp_path, "segmentations", image)
                        segmented = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255
                        segmented = cv2.resize(segmented, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
                        intersection = np.sum(segmented * GOAL_DICT[goal_name])
                        if intersection > max_intersection:
                            max_intersection = intersection
                            best_img = image
                    EXP_RESULTS[f"{folder}_{goal_name}"].append(max_intersection)
                    print("Best img:", best_img)
        for goal_name in GOAL_DICT:
            EXP_RESULTS[f"{folder}_{goal_name}"] = np.mean(EXP_RESULTS[f"{folder}_{goal_name}"])



for val in sorted(EXP_RESULTS):
    print(val)
for val in sorted(EXP_RESULTS):
    print(EXP_RESULTS[val])

pprint(EXP_RESULTS)