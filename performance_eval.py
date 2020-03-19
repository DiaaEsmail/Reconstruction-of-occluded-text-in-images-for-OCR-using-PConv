from __future__ import division

import os, errno
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
VAL_DIR = 'test_toy'#'data_1/images/val'
#TEST_DIR = 'test'#'data_2/images/test'

BATCH_SIZE = 1

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


# Create validation generator
val_datagen = AugmentingDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    MaskGenerator(512, 512, 3),
    target_size=(512, 512),
    batch_size=BATCH_SIZE,
    seed=None
)


# # Instantiate the model

model = PConvUnet()
#model.load("coco_phase2_weights.39-0.36.h5",train_bn=False)
#model.load("coco_phase2_weights.37-0.45.h5",train_bn=False,lr=0.00005)


#rect_2_2 training weights, mask_key = 4
model.load("coco_phase2_weights.50-0.36.h5",train_bn=False, lr=0.00005)

#rect_2, mask_key = 1
#model.load("coco_phase2_weights.43-0.34.h5", train_bn=False, lr=0.00005)


# Store data
ratios = []
psnrs = []
x = 0

output_plot = './permformance_evaluation_2/'
try:
    os.makedirs(output_plot)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
# Loop through test masks released with paper
test_masks = os.listdir('./dia_custom_mask')#os.listdir('./irregular_mask/disocclusion_img_mask')##

for filename in tqdm(test_masks):

    # Load mask from paper
    #filepath = os.path.join('./original_masks_paper/mask/testing_mask_dataset', filename)
    filepath = os.path.join('./dia_custom_mask', filename)
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    mask = cv2.imread(filepath) / 255
    #mask = resized = cv2.resize(mask, (512,512), interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_plot,'mask_{}.jpg'.format(pred_time)),mask*255)
    ratios.append(mask[:,:,0].sum() / (512 * 512))
    mask = np.array([1-mask for _ in range(BATCH_SIZE)])



    # Pick out image from test generator
    test_data = next(val_generator)
    (_, _), ori = test_data
    masked = deepcopy(ori)
    masked[mask==0] = 1





    #(masked, mask), ori = test_data
    for n in range(len(ori)):
        _,axes = plt.subplots(1,3,figsize=(20,5))
        axes[0].imshow(masked[n,:,:,:])
        axes[1].imshow(mask[n,:,:,:]* 1.)
        axes[2].imshow(ori[n,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Mask')
        axes[2].set_title('Raw Image')
        #axes[0].xaxis.set_major_formatter(NullFormatter())
        #axes[1].xaxis.set_major_formatter(NullFormatter())
        #axes[2].xaxis.set_major_formatter(NullFormatter())

        #plt.show()
        plt.savefig(output_plot + 'girl{}_{}.jpg'.format(n, pred_time))

        cv2.imwrite(os.path.join(output_plot, 'masked_input_{}_{}.jpg'.format(n, pred_time) ),masked[n,:,:,:]*255)




    # Run prediction on image & mask
    pred = model.predict([ori, mask])

    for i in range(len(ori)):
        _, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(pred[i,:,:,:] * 1.)
        axes[2].imshow(ori[i,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Raw Image')
        #axes[0].xaxis.set_major_formatter(NullFormatter())
        #axes[0].yaxis.set_major_formatter(NullFormatter())
        #axes[1].xaxis.set_major_formatter(NullFormatter())
        #axes[1].yaxis.set_major_formatter(NullFormatter())

        plt.savefig(output_plot + 'PRED_img_{}_{}.png'.format(i, pred_time))
        plt.close()
        x += 1

    # Only create predictions for about 100 images
    if x > 100:
        break

    # Calculate PSNR
    psnrs.append(-10.0 * np.log10(np.mean(np.square(pred - ori))))



df = pd.DataFrame({'ratios': ratios[:], 'psnrs': psnrs})
export_csv = df.to_csv (output_plot + 'export_dataframe.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
print("\n===== df_raios & psnrs ====:")
print (df)

means, stds = [], []
idx1 = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
idx2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

for mi, ma in zip(idx1, idx2):
    means.append(df[(df.ratios >= mi) & (df.ratios <= ma)].mean())
    stds.append(df[(df.ratios >= mi) & (df.ratios <= ma)].std())


df_index = pd.DataFrame(means, index=['{}-{}'.format(a, b) for a, b in zip(idx1, idx2)])
export_csv = df.to_csv (output_plot + 'mean_dataframe.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
#print("\n===== df_index ====:\n")
#print (df_index)
