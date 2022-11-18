import cv2 as cv
from skimage.morphology import skeletonize
from utils import find_file, auto_canny, show_image, adjust_gamma, f
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def nothing(x):
    pass

def remove_small_cc(*args):
    print(args)
    
def DoG(img, ksize, s1, s2):
    low = cv.GaussianBlur(img, ksize, s1)
    high = cv.GaussianBlur(img, ksize, s2)

    return low - high

#https://stackoverflow.com/questions/65666507/local-contrast-enhancement-for-digit-recognition-with-cv2-pytesseract
def gain_division(img, ksize):
    # Get local maximum:
    maxKernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
    localMax = cv.morphologyEx(img, cv.MORPH_CLOSE, maxKernel, None, None, 1, cv.BORDER_REFLECT101)

    # Perform gain division
    gainDivision = np.where(localMax == 0, 0, (img/localMax))

    # Clip the values to [0,255]
    gainDivision = np.clip((255 * gainDivision), 0, 255)

    # Convert the mat type from float to uint8:
    gainDivision = gainDivision.astype("uint8")

    return gainDivision

def IASF(img, window_size):
    #gˆ(i, j) = a × gr(i, j) + b × g(i, j)
    #gr(i, j) represents the output of the improved range filter
    #a and b are the normalized weighted factors of the improved range filter 
    #and the average filter respectively

    return


matplotlib.use('TKAgg')

image_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni'
mask_path = '/home/alebrasi/Documents/tesi/segmentate_prof/'

mask_extension = 'bmp'
image_extension = 'jpg'

image_name = '109R'


mask_path = find_file(mask_path, f'{image_name}.{mask_extension}')
img_path = find_file(image_path, f'{image_name}.{image_extension}')

print(f'Image path: {img_path}')
print(f'Mask path: {mask_path}')

#img = cv.imread('/home/alebrasi/Documents/tesi/segmentate_prof/Sessione 1/Sessione 1.1/109.bmp')

img = cv.imread(img_path)
mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)

ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.morphologyEx(mask, cv.MORPH_DILATE, ker)

segmented = cv.bitwise_and(img, img, mask=mask)

#segmented = segmented.astype(np.float32)

vals = segmented[segmented > 0]

print(np.median(vals))
print(np.std(vals))
print(np.max(vals))
print(np.min(vals))

print(np.percentile(vals, 95))
print(np.percentile(vals, 5))

segmented = cv.bilateralFilter(segmented, 9, 25, 25)

f_img = f(segmented, alpha=3.0, beta=-450)

g_img = adjust_gamma(f_img, 0.35)

#g_img = cv.GaussianBlur(g_img, (3, 3), 10)

ker = np.array([[0, -1, 0],
                [-1, 5,-1],
                [0, -1, 0]])

#g_img = cv.filter2D(g_img, -1, ker)

#hls = cv.cvtColor(g_img, cv.COLOR_BGR2HLS)
#h,l,s = cv.split(hls)

l1 = cv.cvtColor(g_img, cv.COLOR_BGR2GRAY)

clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(30, 30))
l = clahe.apply(l1)

show_image([l1, l])
#l = gain_division(l, 12)

#show_image([l, g_img[..., ::-1], thr])

#canny = auto_canny(s, mask)

#dxdy = cv.Sobel(s, cv.CV_32F, 0, 1)

#show_image([segmented[..., ::-1], s, dxdy])


"""
cv.namedWindow('Window')

cv.createTrackbar('Sigma1', 'Window', 0, 500, nothing)
cv.createTrackbar('Sigma2', 'Window', 0, 500, nothing)
#cv.createTrackbar('Ksize', 'Window', 3, 21, nothing)
"""

#clahe = cv.createCLAHE(clipLimit=0.1)
#l = clahe.apply(l) 


#_, thr = cv.threshold(l, 0, 255, cv.THRESH_OTSU)

thr = cv.adaptiveThreshold(l, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 0)

_, labels, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(thr, 4, cv.CV_16U, cv.CCL_DEFAULT)

for label, stat in enumerate(stats):
    if stat[cv.CC_STAT_AREA] < 25:
        labels[labels == label] = 0

labels[labels > 0] = 255

labels = labels.astype(np.uint8)

ker = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

morph = cv.morphologyEx(labels, cv.MORPH_OPEN, ker)

#blur = cv.bilateralFilter(l, 5, 30, 30) 
#res = cv.ximgproc.anisotropicDiffusion(np.dstack((l, l, l)), 0.0075, 250, 1000)
show_image([l, g_img[..., ::-1], labels, morph])

"""
while True:
    cv.imshow('Window', dog)

    s1 = cv.getTrackbarPos('Sigma1', 'Window')
    s2 = cv.getTrackbarPos('Sigma2', 'Window')
    #ksize = cv.getTrackbarPos('Ksize', 'Window')

    dog = DoG(l, ksize, s1, s2)

    canny = cv.Canny(s, thr1, thr2)

    _, labels, stats, _ = cv.connectedComponentsWithStats(canny)

    for label, stat in enumerate(stats):
        if stat[cv.CC_STAT_AREA] < 25:
            labels[labels == label] = 0

    labels[labels > 0] = 255


    k = cv.waitKey(20)
    if k == 27:
        break

cv.destroyAllWindows()
"""