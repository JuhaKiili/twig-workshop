import os
from tensorflow.keras import models
import argparse
import cv2
import numpy as np

def prediction_to_text(prediction):
    if prediction[0] > prediction[1]:
        return "It's a CAT! (%s%%)" % round(prediction[0] * 100, 2)
    else:
        return "It's a DOG! (%s%%)" % round(prediction[1] * 100, 2)

parser = argparse.ArgumentParser()
parser.add_argument('-image_size', type=int, default=128, help="Image size")
args = parser.parse_args()

IMAGES_PATH = os.getenv('VH_INPUTS_DIR', '/work') + '/images'
MODEL_PATH = os.getenv('VH_INPUTS_DIR', '/work') + '/model'
IMAGE_SIZE = args.image_size

for path in os.listdir(path=MODEL_PATH):
    path_to_model = os.path.join(MODEL_PATH, path)
    new_model = models.load_model(path_to_model)

for img in os.listdir(path=IMAGES_PATH):
    if img.endswith('jpg') or img.endswith('png') or img.endswith('JPG') or img.endswith('PNG'):
        path_to_img = os.path.join(IMAGES_PATH, img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_COLOR),(args.image_size,args.image_size))
        new_predictions = new_model.predict(img.reshape(-1, args.image_size, args.image_size, 3).astype(float))
        print(path_to_img, " --- ", prediction_to_text(new_predictions[0]))

