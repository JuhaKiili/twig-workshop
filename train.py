import os
import stat
import json
import shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2
from random import shuffle
from tqdm import tqdm
from datetime import datetime
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class EpochCallback(callbacks.Callback):
    best_accuracy = 0.0
    def on_epoch_end(self, epoch, logs={}):
        print(json.dumps({
            'epoch':epoch,
            'training_accuracy': str(logs['accuracy']),
            'training_loss': str(logs['loss']),
            'validated_accuracy': str(logs['val_accuracy']),
            'validated_loss': str(logs['val_loss'])
            }))
        if EpochCallback.best_accuracy < logs['val_accuracy']:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            filepath = MODEL_DIR + '/model-%s-acc-%s-size-%s.h5' % (datetime.now().strftime("%Y%m%d-%H%M%S"), str(logs['val_accuracy']), args.image_size)
            model.save(filepath)
            EpochCallback.best_accuracy = logs['val_accuracy'] 

tf.random.set_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('-learning_rate', type=float, default=0.001, help="Learning rate")
parser.add_argument('-image_size', type=int, default=50, help="Image size")
parser.add_argument('-filter_count', type=int, default=32, help="Filter count")
parser.add_argument('-dense_size', type=int, default=1024, help="Dense size")
parser.add_argument('-epochs', type=int, default=1, help="Epochs")
parser.add_argument('-batch_size', type=int, default=500, help="Batch size")
parser.add_argument('-images_count', type=int, default=100000, help="Image count limit")
parser.add_argument('-validation_count', type=int, default=500, help="Validation count")
parser.add_argument('-rotation', type=float, default="10", help="Augmented rotation")
parser.add_argument('-shear', type=float, default="0.1", help="Augmented shear")
parser.add_argument('-zoom', type=float, default="0.2", help="Augmented zoom")
parser.add_argument('-shift', type=float, default="0.1", help="Augmented scale shift")
parser.add_argument('-fill_mode', type=str, default="reflect", help="Augmented fillmode for edges")
args = parser.parse_args()

TRAIN_DIR = os.getenv('VH_REPOSITORY_DIR', '/work') + '/training_data'
MODEL_DIR = os.getenv('VH_OUTPUTS_DIR', '/work') + '/models'   

def label_image(img):
    img_name = img.split(".")[-3]
    if img_name == "cat":
        return [1,0]
    elif img_name == "dog":
        return [0,1]

full_data = []
for img in tqdm(os.listdir(path=TRAIN_DIR,)[:args.images_count]):
    if img.endswith('jpg'):
        img_label = label_image(img)
        path_to_img = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path_to_img,cv2.IMREAD_COLOR),(args.image_size,args.image_size))
        full_data.append([img,img_label])