import cv2 as cv
import numpy as np
from utils import show_image
import math
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def adjust_gamma(img, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype(np.uint8)

	return cv.LUT(img, table)

def f(img, alpha=2.0, beta=-200):
	alpha = np.array([alpha], dtype=np.uint16)
	
	img = np.clip((img * alpha) + beta, 0, 255).astype(np.uint8)
	return img

def remove_cc(img, area_thr, connectivity=8, algorithm=cv.CCL_DEFAULT):
    _, labels, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(img, connectivity, cv.CV_16U, algorithm)

    for label, stat in enumerate(stats):
        if stat[cv.CC_STAT_AREA] < area_thr:
            labels[labels == label] = 0

    labels[labels > 0] = 255

    return labels.astype(np.uint8)

def clahe_bgr(img, clip, grid_size):
    # TODO: vedere se tenere LAB o cambiare in HLS
    hls = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    h, l, s = cv.split(img)

    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=grid_size)

    l = clahe.apply(s)

    return l

    #return cv.cvtColor(cv.merge([h, l, s]), cv.COLOR_HLS2BGR)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    gray = gray[gray > 0]

    # Calculate grayscale histogram
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = f(image, alpha=alpha, beta=beta)

    #auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

# TODO: Usare solo (rows, cols) o (h, w). 
# Filtrare i risultati con houghlines
def locate_seed_line(img):
    """
    Returns the straighten up image and the extreme points of the seed line

    Parameters:
    img: the image

    Returns:
    img: the straighten up image
    left_pt: a tuple containing the left point of the seed line in (x, y) format
    right_pt: a tuple containing the right point of the seed line in (x, y) format
    """
    #img = cv.bitwise_not(img)##.astype(np.uint32)
    """
    img = img.astype(np.uint32) + 120
    img = np.clip(img, 0, 255).astype(np.uint8)
    """

    SEED_LINE_OFFSET_PX = -5

    #show_image(img[..., ::-1])
    orig_img = img.copy()
    
    img = ~cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (3, 3), 0)
    thresholded_img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, -2)
    # Find horizontal lines with width >= 10px and height = 1px
    horizontal_ker = cv.getStructuringElement(cv.MORPH_RECT, (10, 1))
    horizontal_lines = cv.morphologyEx(thresholded_img, cv.MORPH_OPEN, horizontal_ker)
    #vertical_ker = cv.getStructuringElement(cv.MORPH_RECT, (1, 50))
    #vertical_lines = cv.morphologyEx(img, cv.MORPH_OPEN, vertical_ker)

    """
    Morphological closing for joining nearby (on the y axis) horizontal lines
    This nearby lines are found in correspondence of the edge of the glass slide
    This is done in order to detect them and later removed
    """
    ker = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    horizontal_lines = cv.morphologyEx(horizontal_lines, cv.MORPH_CLOSE, ker)
    
    """
    horizontal_ker = cv.getStructuringElement(cv.MORPH_RECT, (10, 4))
    vertical_ker = cv.getStructuringElement(cv.MORPH_RECT, (4, 20))
    horizontal_lines = cv.morphologyEx(horizontal_lines, cv.MORPH_CLOSE, horizontal_ker)
    vertical_lines = cv.morphologyEx(vertical_lines, cv.MORPH_CLOSE, vertical_ker)
    """

    _, labels, stats, _ =  cv.connectedComponentsWithStatsWithAlgorithm(horizontal_lines, 8, cv.CV_16U, cv.CCL_DEFAULT)

    # Removing thick or short (horizontal) lines
    for label, stat in enumerate(stats):
        w = stat[cv.CC_STAT_WIDTH]
        h = stat[cv.CC_STAT_HEIGHT]
        if w < 40 or h > 6:
            labels[labels == label] = 0

    labels[labels > 0] = 255
    labels = labels.astype(np.uint8)

    show_image(labels)

    horizontal_lines = labels

    h, w = horizontal_lines.shape[:2]

    contours, hierarchy = cv.findContours(horizontal_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rows,cols = horizontal_lines.shape[:2]
    # Find the line that fits the set of horizontal lines
    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
    alpha = math.degrees(math.atan(vy/vx))
    #print(f'{alpha}')

    q = y - ((vy/vx)*x) 
    m = vy/vx

    left_x = 1 
    right_x = 499
    # y = mx + q
    left_y = int((m * left_x) + q)
    right_y = int((m * right_x) + q)

    #cv.line(img1, (left_x, left_y), (right_x, right_y),(0, 255, 0), 1)
    #show_image(img1)

    # Finding the rotation matrix and apply it in order to straighten up the image
    M = cv.getRotationMatrix2D((h//2, w//2), alpha, 1.0)
    img = cv.warpAffine(orig_img, M, (h, w))
    img1 = img.copy()

    #show_image(img1)

    left_pt = np.array([left_x, left_y, 1])
    right_pt = np.array([right_x, right_y, 1])

    left_pt = np.int16(M@left_pt.T)
    right_pt = np.int16(M@right_pt.T)

    left_pt_x, new_y = left_pt[:2]
    new_y = new_y + SEED_LINE_OFFSET_PX

    left_pt = (left_pt[0], new_y)
    right_pt = (right_pt[0], new_y)

    """
    cv.line(img1, left_pt, (right_pt[0], new_y), (255, 0, 0), 1)

    seed_line = np.zeros_like(img)
    cv.line(seed_line, left_pt, right_pt, 255, 1)
    show_image(img1)

    """
    """
    ker = np.array([[-1, -1, 0], 
                    [1, 1, -1],
                    [-1, -1, 0]])

    ker2 = np.flip(ker)

    res = cv.morphologyEx(horizontal_lines, cv.MORPH_HITMISS, ker)
    res2 = cv.morphologyEx(horizontal_lines, cv.MORPH_HITMISS, ker2)

    res = res + res2
    show_image(res)
    """

    return img, left_pt, right_pt 