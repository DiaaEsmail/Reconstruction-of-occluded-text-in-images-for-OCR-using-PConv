from __future__ import division

import os, errno
import random
import gc
import copy
import numpy as np
import pandas as pd
import datetime
import cv2
from PIL import Image
from copy import deepcopy
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, ZeroPadding2D
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras import backend as K
from keras.utils import Sequence
#from keras_tqdm import TQDMNotebookCallback
from keras_tqdm import TQDMCallback

from matplotlib.ticker import NullFormatter
from IPython.display import clear_output

# Import modules from libs/ directory
from pconv_layer import PConv2D
from pconv_model import PConvUnet
from util_pconv import MaskGenerator
#from mask_gen_2 import MaskGenerator
#Setting
TRAIN_DIR = 'coco_train_2017'#'train'#'data/images/train'
VAL_DIR = 'coco_val_2017'#'val'#'data_1/images/val'
TEST_DIR = 'coco_test_2017'#'test'#'data_2/images/test'

BATCH_SIZE = 6
#key = random.randint(1,6)
#print("key value: {}".format(key))
#Creating train & test data generator
class AugmentingDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:

            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample
            mask = np.stack([
                mask_generator.sample(seed)
                for _ in range(ori.shape[0])], axis=0
            )

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori


# Create training generator
train_datagen = AugmentingDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    MaskGenerator(512, 512,3,None),
    target_size=(512, 512),
    batch_size=BATCH_SIZE,
    seed = None
)

# Create validation generator
val_datagen = AugmentingDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    MaskGenerator(512, 512, 3, None),
    target_size=(512, 512),
    batch_size=BATCH_SIZE,
    seed=None
)


# Create testing generator
test_datagen = AugmentingDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    MaskGenerator(512, 512,3,None),
    target_size=(512, 512),
    batch_size=BATCH_SIZE,
    seed=None
)


output_dir = './phase_1_coco_2017_output/test_samples'
try:
    os.makedirs(output_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
# Pick out an example
test_data = next(test_generator)
(masked, mask), ori = test_data
# Show side by side
for i in range(len(ori)):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[i,:,:,:])
    axes[1].imshow(mask[i,:,:,:] * 1.)
    axes[2].imshow(ori[i,:,:,:])
    plt.show()
    plt.savefig(output_dir + '/img_{}.jpg'.format(i))


#training
def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""

    output_plot = './phase_1_coco_2017_output_plot/plot_samples'
    try:
        os.makedirs(output_plot)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # Get samples & Display them
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(pred_img[i,:,:,:] * 1.)
        axes[2].imshow(ori[i,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')

        plt.savefig(output_plot + '/img_{}_{}.png'.format(i, pred_time))
        plt.close()


#Phase 1 -with batch BatchNormalizationN
# Instantiate the model
model = PConvUnet(vgg_weights='vgg16_pytorch2keras.h5')
model.load("coco_2017_weights.10-1.48.h5")

FOLDER = './phase_1_coco_2017_data_log/logs/coco_phase1'
# # Run training for certain amount of epochs
model.fit_generator(
    train_generator,
    steps_per_epoch=10000,
    validation_data=val_generator,
    validation_steps=1000,
    epochs=50,
    verbose=1,
    callbacks=[
        TensorBoard(
            log_dir=FOLDER,
            write_graph=False
        ),
        ModelCheckpoint(
            FOLDER+'coco_2017_phase_1_weights.{epoch:02d}-{loss:.2f}.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: plot_callback(model)
        ),
        TQDMCallback()
    ]
)
