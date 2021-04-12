import tensorflow as tf
from PIL import Image
import numpy as np
import imutils
import cv2
import os

list = os.listdir('./test/') 
list.sort()

for file in list:
    if file.endswith('jpg') or file.endswith('png')or file.endswith('JPG'):
        img = cv2.imread('./test/' + file, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('test/test/test.jpg'+file, img)