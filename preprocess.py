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

def locate_seed_line(img):
    #img = cv.bitwise_not(img)##.astype(np.uint32)
    """
    img = img.astype(np.uint32) + 120
    img = np.clip(img, 0, 255).astype(np.uint8)
    """
    show_image(img[..., ::-1])

    img = ~cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, -2)
    show_image(img)
    horizontal_ker = cv.getStructuringElement(cv.MORPH_RECT, (10, 1))
    vertical_ker = cv.getStructuringElement(cv.MORPH_RECT, (1, 50))
    horizontal_lines = cv.morphologyEx(img, cv.MORPH_OPEN, horizontal_ker)
    vertical_lines = cv.morphologyEx(img, cv.MORPH_OPEN, vertical_ker)

    show_image(horizontal_lines)

    ker = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    horizontal_lines = cv.morphologyEx(horizontal_lines, cv.MORPH_CLOSE, ker)

    show_image(horizontal_lines)

    """
    horizontal_ker = cv.getStructuringElement(cv.MORPH_RECT, (10, 4))
    vertical_ker = cv.getStructuringElement(cv.MORPH_RECT, (4, 20))
    horizontal_lines = cv.morphologyEx(horizontal_lines, cv.MORPH_CLOSE, horizontal_ker)
    vertical_lines = cv.morphologyEx(vertical_lines, cv.MORPH_CLOSE, vertical_ker)
    """

    horizontal_lines = remove_cc(horizontal_lines, 100)

    _, labels, stats, _ =  cv.connectedComponentsWithStatsWithAlgorithm(horizontal_lines, 8, cv.CV_16U, cv.CCL_DEFAULT)

    for label, stat in enumerate(stats):
        if stat[cv.CC_STAT_AREA] > 250:
            labels[labels == label] = 0

    labels[labels > 0] = 255

    labels = labels.astype(np.uint8)

    show_image(labels)

    horizontal_lines = labels

    h, w = horizontal_lines.shape[:2]

    contours, hierarchy = cv.findContours(horizontal_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rows,cols = horizontal_lines.shape[:2]
    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)

    print(f'{vx}, {vy}, {x}, {y}')

    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    tmp = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.line(tmp,(cols-1,righty),(0,lefty),(0,255,0),1)

    show_image(img)

    ker = np.array([[-1, -1, 0], 
                    [1, 1, -1],
                    [-1, -1, 0]])

    ker2 = np.flip(ker)

    res = cv.morphologyEx(horizontal_lines, cv.MORPH_HITMISS, ker)
    res2 = cv.morphologyEx(horizontal_lines, cv.MORPH_HITMISS, ker2)

    res = res + res2
    show_image(res)

    return