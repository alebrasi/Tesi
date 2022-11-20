import cv2 as cv
from skimage.morphology import skeletonize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from crf import CRF
from preprocess import adjust_gamma, automatic_brightness_and_contrast, clahe_bgr, remove_cc
from utils import find_file, show_image, f

matplotlib.use('TKAgg')

image_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni'
mask_path = '/home/alebrasi/Documents/tesi/segmentate_prof/'

mask_extension = 'bmp'
image_extension = 'jpg'

image_name = '109'


mask_path = find_file(mask_path, f'{image_name}.{mask_extension}')
img_path = find_file(image_path, f'{image_name}.{image_extension}')


print(f'Image path: {img_path}')
print(f'Mask path: {mask_path}')

#img = cv.imread('/home/alebrasi/Documents/tesi/segmentate_prof/Sessione 1/Sessione 1.1/109.bmp')

img = cv.imread(img_path)
mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)

# Chiusura morfologica sulla maschera
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.morphologyEx(mask, cv.MORPH_DILATE, ker)

segmented = cv.bitwise_and(img, img, mask=mask)

#segmented = cv.bilateralFilter(segmented, 5, 5, 100)

beta = -400

f_img = f(segmented, alpha=3.0, beta=beta)


f1_img, alpha, beta = automatic_brightness_and_contrast(segmented, 50)
f_img = adjust_gamma(f_img, 0.35)

#f1_img = adjust_gamma(f1_img, 0.38)

clahe_img = clahe_bgr(f1_img, 1, (20, 20))

print(f'{alpha}, {beta}')

show_image([f_img[..., ::-1], f1_img[..., ::-1], clahe_img[..., ::-1]])

f1_img = clahe_img

a1 = CRF(mask, f1_img, 5)
a = CRF(mask, f_img, 5)


a2 = np.zeros_like(a1)
a2 = a1[..., 1]

cc_rem = remove_cc(a2, 20, connectivity=4)

show_image([img, (a, "manual"), (a1, "auto"), (cc_rem, "cc")])