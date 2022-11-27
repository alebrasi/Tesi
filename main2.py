import cv2 as cv
from skimage.morphology import skeletonize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bm3d

from crf import CRF
from preprocess import adjust_gamma, automatic_brightness_and_contrast, clahe_bgr, remove_cc, locate_seed_line
from utils import find_file, show_image, f

matplotlib.use('TKAgg')

image_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni'
mask_path = '/home/alebrasi/Documents/tesi/segmentate_prof/'
mask_extension = 'bmp'
image_extension = 'jpg'

image_name = '950'

mask_path = find_file(mask_path, f'{image_name}.{mask_extension}')
img_path = find_file(image_path, f'{image_name}.{image_extension}')

print(f'Image path: {img_path}')
print(f'Mask path: {mask_path}')

#img = cv.imread('/home/alebrasi/Documents/tesi/segmentate_prof/Sessione 1/Sessione 1.1/109.bmp')

img = cv.imread(img_path)

locate_seed_line(img)
"""
mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)

# Morphological closing for filling small gaps in the mask
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker)

segmented = cv.bitwise_and(img, img, mask=mask)

f_img, alpha, beta = automatic_brightness_and_contrast(segmented, 50) # TODO: Sperimentare sul clip limit

clahe_img = clahe_bgr(f_img, 1, (30, 30)) 

# Gamma adjustment
l  = adjust_gamma(clahe_img, 1.5)
# Removing noise due to gamma adjustment
_, l = cv.threshold(l, 25, 255, cv.THRESH_TOZERO)

#show_image([clahe_img, l])

thr = cv.adaptiveThreshold(l, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 0)
cc_rem = remove_cc(thr, 50)

# Blurring for smoothing the thresholded image
blurred = cv.GaussianBlur(cc_rem, (3,3), 5) # TODO: Sostituibile con box filter?

# Thresholding the blurred image in order to obtain a smoother segmentation
cc_rem = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 0)

show_image([blurred, cc_rem])
# Mask normalization for skeletonize
cc_rem[cc_rem == 255] = 1
skeleton = skeletonize(cc_rem)

show_image([(l, 'Grayscale immagine rifinita'), (cc_rem, 'threshold'), (skeleton, 'skeleton maschera rifinita')])
"""