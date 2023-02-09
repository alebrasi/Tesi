import cv2 as cv
import numpy as np

from preprocess import automatic_brightness_and_contrast, adjust_gamma, f
from utils import show_image

def refine_region_below(img, dbg_ctx):
    f_img, alpha, beta = automatic_brightness_and_contrast(img, 37)
    show_image((f_img, 'automatic brightness and contrast'), dbg_ctx=dbg_ctx)

    # Noise removes
    blur = cv.pyrDown(f_img)
    blur = cv.pyrUp(blur)
    f_img = blur

    # CLAHE equalization
    lab = cv.cvtColor(f_img, cv.COLOR_BGR2LAB)
    l,a,b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=40.0, tileGridSize=(20, 20))
    clahe_img = clahe.apply(l)

    show_image([(l, 'luminosità'), (clahe_img, 'clahe')], dbg_ctx=dbg_ctx)

    _, l = cv.threshold(clahe_img, 50, 255, cv.THRESH_TOZERO)
    show_image(l)

    thr = cv.adaptiveThreshold(
        l, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 0)
    
    #thr = cv.pyrUp(thr)
    #thr = cv.adaptiveThreshold(
    #    thr, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 0)

    show_image((thr, 'thresholded'), dbg_ctx=dbg_ctx)

    return thr

def refine_region_above(img, dbg_ctx):
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
    morph = cv.morphologyEx(img, cv.MORPH_CLOSE, ker)
    return morph

def locate_seeds(img, dbg_ctx):
    h, w = img.shape[:2]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    show_image(hsv, cmap='magma', dbg_ctx=dbg_ctx)

    #res = cv.inRange(hsv, (15, 80, 100), (35, 255, 255))
    res = cv.inRange(hsv, (15, 100, 100), (25, 255, 255))
    show_image([res, img[..., ::-1]], dbg_ctx=dbg_ctx)

    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    morphed = cv.morphologyEx(res, cv.MORPH_OPEN, ker)
    # TODO: Dilate o close?

    #ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
    morphed = cv.morphologyEx(morphed, cv.MORPH_DILATE, ker)

    _, labels, stats, _ = cv.connectedComponentsWithStats(morphed)

    candidate_seeds_box = []
    yy, xx = np.mgrid[:h, :w]

    for i, stat in enumerate(stats[1:]):
        i = i+1
        if stat[cv.CC_STAT_AREA] < 200:
            labels[i] = 0
        else:
            x, y = stat[cv.CC_STAT_LEFT], stat[cv.CC_STAT_TOP]
            width, height = stat[cv.CC_STAT_WIDTH], stat[cv.CC_STAT_HEIGHT]
            xs = xx[labels == i]
            ys = yy[labels == i]
            pos = np.column_stack((xs, ys))

            #rect = cv.minAreaRect(pos)
            #box = np.int0(cv.boxPoints(rect))
            #center, _, _ = rect
            candidate_seeds_box.append((x, y, width, height))

            print(f'Seed {i} BB: ')
            print(f'\t x: {x}')
            print(f'\t y: {y}')
            print(f'\t w: {width}')
            print(f'\t h: {height}')

            cv.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 1)
            #cv.drawContours(img, [box], 0, (0, 0, 255), 1)
            #cx, cy = np.int0(center)
            #img[cy, cx] = [255, 0, 0]

    show_image(labels, cmap='magma', dbg_ctx=dbg_ctx)
    show_image(img[..., ::-1], dbg_ctx=dbg_ctx)

    return candidate_seeds_box