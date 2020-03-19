from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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

#Setting
#TRAIN_DIR = 'train'#'data/images/train'
#VAL_DIR = 'val'#'data_1/images/val'
TEST_DIR = 'synth_testset'#'data_2/images/test'
BATCH_SIZE = 1

import os, errno
import sys


def get_name(file):
    base = os.path.basename(file)
    name = os.path.splitext(base)[0]

    return name

def loadImages(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


#Creating test data generator
class AugmentingDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']
        index = 0
        while True:

            seed = random.randint(0,42)
            # base = os.path.basename(directory)
            # name = os.path.splitext(base)[0]
            # Get augmentend image samples
            ori = next(generator)
            try:
                path = generator._filepaths[index]
            except:
                break
            index += 1

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
            #yield [masked, mask], ori,name
            yield [masked, mask], ori, path



# Create testing generator
test_datagen = AugmentingDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    MaskGenerator(512, 512, 3),
    target_size=(512, 512),
    batch_size=BATCH_SIZE,
    seed=42, shuffle =False
)




# Instantiate the model
model = PConvUnet()
#model.load("coco_phase2_weights.37-0.45.h5",train_bn=False,lr=0.00005)
model.load("coco_phase2_weights.01-0.37.h5",train_bn=False)



#Generating samples
output_plot = './predicted_synth_dataset/pred_samples_bs1'
try:
    os.makedirs(output_plot)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

n = 0


for (masked, mask), ori, path in tqdm(test_generator):
    name = os.path.basename(path)
    print(path)
    #Run predictions for this batch of new_images
    pred_img = model.predict([masked,mask],batch_size=BATCH_SIZE)
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #cv2.imwrite('{}.png'.format(n),ori)
    #file_name =  "{}.jpg".format(n)
    #cv2.imwrite(os.path.join(output_plot,file_name),ori[n,:,:,:] * 255.)
    for i in range(len(masked)):
        cv2.imwrite(os.path.join(output_plot,name),pred_img[i,:,:,:] * 255.)
        print("Batches {}/{}".format(i,len(ori)))
        print("n {}".format(n))
        sys.stdout.flush()
        n += 1



    # #Clear current output and display test images
    # for i in range(len(ori)):
    #     _, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     axes[0].imshow(masked[i,:,:,:])
    #     axes[1].imshow(pred_img[i,:,:,:] * 1.)
    #     axes[0].set_title('Masked Image')
    #     axes[1].set_title('Predicted Image')
    #     axes[0].xaxis.set_major_formatter(NullFormatter())
    #     axes[0].yaxis.set_major_formatter(NullFormatter())
    #     axes[1].xaxis.set_major_formatter(NullFormatter())
    #     axes[1].yaxis.set_major_formatter(NullFormatter())
    #
    #
    #     #plt.savefig(output_plot + '/img_{}_{}.png'.format(i, pred_time))
    #
    #     #file_name =  "{}.jpg".format(i)
    #     #file_name =  name[n] + ".jpg"
    #     #cv2.imwrite(os.path.join(output_plot,file_name),ori[i,:,:,:] * 255.)
    #
    #     #file_name =  names_image[i] + ".jpg"
    #     cv2.imwrite(os.path.join(output_plot,name),pred_img[i,:,:,:] * 255.)
    #
    #     plt.close()
    #
    #     print("Batches {}/{}".format(i,len(ori)))
    #     print("n {}".format(n))
    #
    #     sys.stdout.flush()
    #
    #     n += 1






    #Only create predictions for about 100 images
    #if n > 21:
    #    break
