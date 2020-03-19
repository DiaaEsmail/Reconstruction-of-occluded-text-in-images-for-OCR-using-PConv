import os
import random
import gc
import copy
import numpy as np
from PIL import Image
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import datetime

from keras.layers import Input, Dense, ZeroPadding2D
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

# Import modules from libs/ directory
from pconv_layer import PConv2D
from pconv_model import PConvUnet
#from mask_gen_2 import MaskGenerator
from util_pconv import MaskGenerator

'''
   The paper uses U-Net architecture for doing the image inpainting. '''

# Settings
BATCH_SIZE = 6
key = random.randint(1,6)
#print(key)

# Imagenet Rescaling
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

#PConvUnet().summary()




# Instantiate mask generator
#mask_generator = MaskGenerator(512, 512,key, 3, rand_seed=None)
mask_generator = MaskGenerator(512, 512, 3, rand_seed=None)



# Load image
img = np.array(Image.open('000000000009.jpg').resize((512,512))) / 255

#Load mask
mask = mask_generator.sample()


# Image + mask
masked_img = deepcopy(img)
masked_img[mask==0] = 1


# Show side by side original_image - mask - masked_image
_, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(img)
axes[1].imshow(mask*255)
axes[2].imshow(masked_img)
#plt.show()
plt.savefig('coco_2017_masked_img.jpg')


#Creating data generator
def plot_sample_data(masked, mask, ori, middle_title='Raw Mask'):
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[:,:,:])
    axes[0].set_title('Masked Input')
    axes[1].imshow(mask[:,:,:])
    axes[1].set_title(middle_title)
    axes[2].imshow(ori[:,:,:])
    axes[2].set_title('Target Output')
    #plt.show()
    plt.savefig('coco_2017_sample_data_{}.jpg'.format(pred_time))



'''
   In this simple testing case, we'll only be testing
   the architecture  on a singel image to see how it
   performs. We create a generator that will infinitely yield
   the same image and masked_image for us, except each yielded
   image will be slightly augmented using ImageDataGenerator from
   keras.processing'''

class DataGenerator(ImageDataGenerator):


    def flow(self, x, *args, **kwargs):
        while True:
            # Get augmentend image samples
            ori = next(super().flow(x, *args, **kwargs))

            # Get masks for each image sample
            mask = np.stack([mask_generator.sample() for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = copy.deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori



# Create datagen
datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# Create generator from numpy array
batch = np.stack([img for _ in range(BATCH_SIZE)], axis=0)
generator = datagen.flow(x=batch, batch_size=BATCH_SIZE)


[m1, m2], o1 = next(generator)
plot_sample_data(m1[0], m2[0]*255, o1[0])

#Training the inpainting UNet on single image
'''
   Now that we have a generator, we can initiate the training, for
   fitting the fit() function of PConvUnet takes a callback, which we
   can use to evaluate and display the progress in terms of reconstructing
   the targe image based on the masked input image.'''

# Instantiate model
model = PConvUnet(vgg_weights='vgg16_pytorch2keras.h5')
#model.load("weights.09-0.67.h5")

model.fit_generator(
    generator,
    steps_per_epoch=2000,
    epochs=10,
    callbacks=[
        TensorBoard(
            log_dir='./coco_2017_data/logs/single_image_test',
            write_graph=False
        ),
        ModelCheckpoint(
            './coco_2017_data/logs/single_image_test/coco_2017_weights.{epoch:02d}-{loss:.2f}.h5',
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        ),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: plot_sample_data(
                masked_img,
                model.predict(
                    [
                        np.expand_dims(masked_img,0),
                        np.expand_dims(mask,0)
                    ]
                )[0]
                ,
                img,
                middle_title='Prediction'
            )
        )
    ],
)
