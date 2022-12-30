import cv2 as cv
from skimage.morphology import skeletonize
from utils import find_file, auto_canny, show_image, adjust_gamma, f
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
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

def crf(anno_rgb, img, inference_steps):

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_rgb = anno_rgb.astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)


    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    #print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(inference_steps)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    #imwrite(fn_output, MAP.reshape(img.shape))

    return MAP.reshape(img.shape)


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

image_name = '950R'


mask_path = find_file(mask_path, f'{image_name}.{mask_extension}')
img_path = find_file(image_path, f'{image_name}.{image_extension}')


print(f'Image path: {img_path}')
print(f'Mask path: {mask_path}')

#img = cv.imread('/home/alebrasi/Documents/tesi/segmentate_prof/Sessione 1/Sessione 1.1/109.bmp')

img = cv.imread(img_path)
mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)

h, w = mask.shape[0], mask.shape[1]

mask_crf = np.zeros((h, w, 3))
mask_crf[..., 1] = mask
mask_crf[..., 2] = cv.bitwise_not(mask)



ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.morphologyEx(mask, cv.MORPH_DILATE, ker)

segmented = cv.bitwise_and(img, img, mask=mask)

#segmented = segmented.astype(np.float32)

segmented = cv.bilateralFilter(segmented, 9, 50, 50)

f_img = f(segmented, alpha=3.0, beta=-400)

g_img = adjust_gamma(f_img, 0.35)

mask1 = crf(mask_crf, g_img, 300)
show_image([mask, mask1])

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
show_image([l, g_img[..., ::-1], labels, mask1, morph])

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