import cv2 as cv
from skimage.morphology import skeletonize
from utils import find_file, auto_canny, show_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def nothing(x):
    pass

def remove_small_cc(*args):
    print(args)
    

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

segmented = cv.bitwise_and(img, img, mask=mask)

hls = cv.cvtColor(segmented, cv.COLOR_BGR2HLS)
h, l, s = cv.split(hls)

ker = np.array([[0, -1, 0],
                [-1, 5,-1],
                [0, -1, 0]])

s = cv.filter2D(s, -1, ker)

gamma = 1.6 

s = ((s/255.)**(1/gamma)) * 255

s = cv.normalize(s, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

s = cv.GaussianBlur(s, (5, 5), 20)


#canny = auto_canny(s, mask)

dxdy = cv.Sobel(s, cv.CV_32F, 0, 1)

show_image([segmented[..., ::-1], s, dxdy])

"""
canny_inv = cv.bitwise_not(canny)

res = cv.bitwise_and(mask, mask, mask=canny_inv)

ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
gradient = cv.morphologyEx(s, cv.MORPH_GRADIENT, ker)

#_, gradient = cv.threshold(gradient, 10, 255, cv.THRESH_BINARY)
"""

"""
cv.namedWindow('Window')

cv.createTrackbar('Area', 'Window', 0, 100, nothing)
cv.createTrackbar('Thr1', 'Window', 0, 255, nothing)
cv.createTrackbar('Thr2', 'Window', 0, 255, nothing)

while True:
    cv.imshow('Window', canny)

    thr1 = cv.getTrackbarPos('Thr1', 'Window')
    thr2 = cv.getTrackbarPos('Thr2', 'Window')
    area = cv.getTrackbarPos('Area', 'Window')

    #print(area)
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
#_, thr = cv.threshold(labels, 1, 255, cv.NORM_MINMAX)

#show_image(labels)

#show_image([img, s, canny, res])

#plt.imshow(canny, cmap='gray')
#plt.show()