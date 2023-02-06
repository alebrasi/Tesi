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
# https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = image
    if len(image.shape) == 3:
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
    
    """
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    show_image(image)
    """

    auto_result = f(image, alpha=alpha, beta=beta)

    #auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def locate_seed_line(img, rough_location=None, seed_line_offset_px=-10, dbg_ctx=None):
    """
    Returns the rotation matrix that straighten the image and the extreme points of the seed line

    Parameters:
    img: the image
    rough_location (array): rough location of the seed line expressed as an array containing
                    the top-left point and bottom-right point of the ROI in (y, x) format.
    seed_line_offset_px (int): offset to apply to the found line extreme points (left_pt and right_pt)

    Returns:
    M: the rotation matrix
    left_pt: a tuple containing the left point of the seed line in (x, y) format
    right_pt: a tuple containing the right point of the seed line in (x, y) format
    """

    orig_img = img.copy()
    offset_x = offset_y = 0
    img = ~cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    show_image(img, dbg_ctx=dbg_ctx)

    # TODO: Provare diversi tileGridSize
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(60, 60))
    img = clahe.apply(img)
    show_image(img, dbg_ctx=dbg_ctx)
    if rough_location != None:
        top_left_pt, bottom_right_pt = rough_location[0], rough_location[1]
        tl_y, tl_x = top_left_pt
        br_y, br_x = bottom_right_pt
        img = img[tl_y:br_y, tl_x:br_x]
        offset_x, offset_y = tl_x, tl_y
    show_image(img, dbg_ctx=dbg_ctx)
    img = cv.GaussianBlur(img, (5, 5), 0)
    thresholded_img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                            cv.THRESH_BINARY, 13, -2)
    show_image(thresholded_img, dbg_ctx=dbg_ctx)
    # Find horizontal lines with width >= 20px and height = 1px
    horizontal_ker = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
    horizontal_lines = cv.morphologyEx(thresholded_img, cv.MORPH_OPEN, horizontal_ker)
    #vertical_ker = cv.getStructuringElement(cv.MORPH_RECT, (1, 50))
    #vertical_lines = cv.morphologyEx(img, cv.MORPH_OPEN, vertical_ker)

    show_image(horizontal_lines, dbg_ctx=dbg_ctx)
    
    lines = cv.HoughLinesWithAccumulator(horizontal_lines, 1, np.pi / 180, 50, None, 0, 0)
    # obtaining the line with the maximum votes, even if the Hough transform should return 
    # the results in descending order by accumulator value
    lines = lines.reshape(-1, 3)
    line = max(lines, key=lambda l: l[2])

    rho, theta, _ = line

    if dbg_ctx != None and dbg_ctx.is_active:
        cdst = cv.cvtColor(horizontal_lines, cv.COLOR_GRAY2BGR)
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        cv.line(cdst, pt1, pt2, (0, 255, 0), 2, cv.LINE_AA)
        show_image(cdst)
    
    h, w = horizontal_lines.shape[:2]

    rot_angle = math.degrees(theta)-90.0
    print(rot_angle)

    M = cv.getRotationMatrix2D((h//2, w//2), rot_angle, 1.0)

    roi_offset_y, _ = rough_location[0]

    #y = int(rho*math.cos(math.radians(rot_angle))) + roi_offset_y + seed_line_offset_px 
    y = int(rho/math.sin(theta)) + roi_offset_y + seed_line_offset_px 
    
    pt1 = (0, y)
    pt2 = (w-1, y)
    
    return M, pt1, pt2

    """ 
    _, labels, stats, _ =  cv.connectedComponentsWithStatsWithAlgorithm(horizontal_lines, 8, 
                                                                        cv.CV_16U, cv.CCL_DEFAULT)

    show_image(labels, cmap='magma', dbg_ctx=dbg_ctx)
    # Removing thick or short (horizontal) lines
    for label, stat in enumerate(stats):
        w = stat[cv.CC_STAT_WIDTH]
        h = stat[cv.CC_STAT_HEIGHT]
        if w > 45:
            continue
        if w < 35 or h > 4:
            labels[labels == label] = 0

    labels[labels > 0] = 255
    labels = labels.astype(np.uint8)
    show_image(labels, dbg_ctx=dbg_ctx)

    #show_image(cv.bitwise_and(orig_img, orig_img, mask=labels))

    horizontal_lines = labels
    
    h, w = horizontal_lines.shape[:2]

    contours, hierarchy = cv.findContours(horizontal_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rows,cols = horizontal_lines.shape[:2]
    # Find the line that fits the set of horizontal lines
    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
    alpha = math.degrees(math.atan(vy/vx))
    print(f'{alpha}°')
    #x = x + offset_x
    #y = y + offset_y
    q = y - ((vy/vx)*x) 
    m = vy/vx

    left_x = 1 
    right_x = w-1
    # y = mx + q
    left_y = int((m * left_x) + q) + offset_y
    right_y = int((m * right_x) + q) + offset_y

    #print(f'l_x{}')

    cv.line(img, (left_x, left_y), (right_x, right_y),(0, 255, 0), 1)
    show_image(img, dbg_ctx=dbg_ctx)

    # Finding the rotation matrix
    h, w = orig_img.shape[:2]
    M = cv.getRotationMatrix2D((h//2, w//2), alpha, 1.0)
    #img = cv.warpAffine(orig_img, M, (h, w))
    #img1 = img.copy()

    #show_image(img1)

    left_pt = np.array([left_x, left_y, 1])
    right_pt = np.array([right_x, right_y, 1])

    left_pt = np.int16(M@left_pt.T)
    right_pt = np.int16(M@right_pt.T)

    left_pt_x, new_y = left_pt[:2]
    new_y = new_y + seed_line_offset_px

    left_pt = (left_pt[0], new_y)
    right_pt = (right_pt[0], new_y)

    return M, left_pt, right_pt
    """