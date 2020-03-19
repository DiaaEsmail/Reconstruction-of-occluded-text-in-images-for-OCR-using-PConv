import numpy as np
import torchvision.models as models
from keras.applications.vgg16 import VGG16


'''
   Weights of the VGG16 convolutional layers (include_top=False) ported from PyTorch to Keras.


   -Why do you need this?
   The VGG16 model in Keras comes with weights ported from the original Caffe implementation. It expects the following image pre-processing:
   convert the images from RGB to BGR,
   subtract [103.939, 116.779, 123.68] from the BGR channels, respectively.

   -The VGG16 model in PyTorch comes with a different set of weights and expects another pre-processing:
   divide the image by 255,
   subtract [0.485, 0.456, 0.406] from the RGB channels, respectively,
   divide the RGB channels by [0.229, 0.224, 0.225], respectively.

'''

pytorch_model = models.vgg16(pretrained=True)

# select weights in the conv2d layers and transpose them to keras dim ordering:
wblist_torch = list(pytorch_model.parameters())[:26]
wblist_keras = []
for i in range(len(wblist_torch)):
    if wblist_torch[i].dim() == 4:
        w = np.transpose(wblist_torch[i].detach().numpy(), axes=[2, 3, 1, 0])
        wblist_keras.append(w)
    elif wblist_torch[i].dim() == 1:
        b = wblist_torch[i].detach().numpy()
        wblist_keras.append(b)
    else:
        raise Exception('Fully connected layers are not implemented.')

keras_model = VGG16(include_top=False, weights=None)
keras_model.set_weights(wblist_keras)
keras_model.save_weights('vgg16_pytorch2keras.h5')
