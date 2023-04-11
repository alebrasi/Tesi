import cv2 as cv
import numpy as np
from misc.utils import show_image
import math


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)

    return cv.LUT(img, table)


def f(img, alpha=2.0, beta=-200.0):
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

# Automatic brightness and contrast optimization with optional histogram clipping
# https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
def automatic_brightness_and_contrast(image, mask, clip_hist_percent=1):
    gray = image
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mean, std = cv.meanStdDev(gray[mask == 255])
    print(f'Mean: {mean}, Std: {std[0][0]}')

    # hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
    # plt.plot(hist)
    # plt.show()

    # Calculate grayscale histogram
    hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0  # TODO: Provare con 0 o 1
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
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

    return auto_result, alpha, beta


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
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))
        cv.line(cdst, pt1, pt2, (0, 255, 0), 2, cv.LINE_AA)
        show_image(cdst)

    h, w = horizontal_lines.shape[:2]

    rot_angle = math.degrees(theta) - 90.0
    print(rot_angle)

    M = cv.getRotationMatrix2D((h // 2, w // 2), rot_angle, 1.0)

    roi_offset_y, _ = rough_location[0]

    y = int(rho) + roi_offset_y + seed_line_offset_px
    print(theta, " ", rho, " ", y, " ", rot_angle)

    pt1 = (0, y)
    pt2 = (w - 1, y)

    return M, pt1, pt2


def calc_rot_angle(theta, rho, x1, x2):
    calc = lambda rho, theta, x: (rho / math.sin(theta)) - (math.cos(theta) / math.sin(theta)) * x
    y1 = calc(rho, theta, x1)
    y2 = calc(rho, theta, x2)
    return math.degrees(math.atan2(y2 - y1, (x2 - x1)))
