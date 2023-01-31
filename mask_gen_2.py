import os
from random import randint, seed
import itertools
import numpy as np
import cv2
import random


class MaskGenerator():

    def __init__(self, height, width, key, channels=3, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width

        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = height
        self.width = width
        self.channels = channels
        self.key = key
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))

        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")



      
##    if key == 1:
##        img = cv2.circle(resize(img),(center_1,256), radius, (c1,c2,c3), -1)
##    elif key == 2:
##        img =  cv2.line(resize(img),(0,256),(256,256) , (c1,c2,c3), 50)
##    elif key == 3:
##        img = cv2.rectangle(resize(img),(z,250),(400, 450),(c1,c3,c2),-1,10)
##
##    return img

        if self.key == 1:
            cv2.rectangle(img,(1,100),(100,400),(1,1,1),-1,10)
        if self.key == 2:
            cv2.rectangle(img,(100,100),(200,400),(1,1,1),-1,10)
        if self.key == 3:
            cv2.rectangle(img,(200,100),(300,400),(1,1,1),-1,10)
        if self.key == 4:
            cv2.rectangle(img,(300,100),(400,400),(1,1,1),-1,10)
        if self.key == 5:
            cv2.rectangle(img,(400,100),(500,400),(1,1,1),-1,10)
        if self.key == 6:
            cv2.rectangle(img,(355,100),(450,400),(1,1,1),-1,10)

        return 1-img
        # Draw random rectangle
##        for _ in range(1):
##            #x1, x2 = randint(100,200 ), randint(1, self.width)
##            #y1, y2 = randint(1, self.height), randint(1, self.height)
##            thickness = randint(3, size)
##            x1 = random.randint(200, 300)
##            x2 = random.randint(350,450)
##            y1 = random.randint(200,300)
##            y2 = random.randint(350,500)
##            #cv2.rectangle(img,(1,200),(200,400),(1,1,1),-1,10)#-1,10)
##            cv2.rectangle(img,(x1,y1),(x2,y2),(1,1,1),-1,10)#-1,10)
##                             #(x1,y1),(x2,y2)

        #return 1-img





    def _load_mask(self, rotation=True, dilation=True, cropping=False):
        """Loads a mask from disk, and optionally augments it"""

        # Read image
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(self.mask_files, 1, replace=False)[0]))

        # Random rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        # Random cropping
        if cropping:
            #x = np.random.randint(0, mask.shape[1] - self.width)
            #y = np.random.randint(0, mask.shape[0] - self.height)

            x = np.random.randint(100,300)
            y = np.random.randint(150,252)
            mask = mask[y:y+self.height, x:x+self.width]

        return (mask > 1).astype(np.uint8)


    def sample(self, random_seed=None):
        """Retrieve a random mask"""
        if random_seed:
            seed(random_seed)
        if self.filepath and len(self.mask_files) > 0:
            return self._load_mask()
        else:
            return self._generate_mask()
