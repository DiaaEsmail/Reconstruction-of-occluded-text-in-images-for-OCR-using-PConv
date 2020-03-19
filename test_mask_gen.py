import os
import itertools
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from mask_gen_2 import MaskGenerator


key = 4
mask_generator = MaskGenerator(512, 512, key, 3)


img = mask_generator.sample()
#img_rgb = cv2.imread(img)
#Conv_hsv_Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lower_black = np.array([0,0,0], dtype = "uint16")
upper_black = np.array([0,0,0], dtype = "uint16")
black_mask = cv2.inRange(img, lower_black, upper_black)
cv2.imshow('mask0',black_mask)
#img[np.where(img == [255])] = [0]

cv2.imwrite("dia_4_cust_mask.jpg",black_mask)
