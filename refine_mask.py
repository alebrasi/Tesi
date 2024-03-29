import cv2 as cv
import numpy as np

from misc.logger import Logger
from preprocess import automatic_brightness_and_contrast
from misc.utils import show_image


def refine_region_below(img, mask, dbg_ctx):
    f_img, alpha, beta = automatic_brightness_and_contrast(img, mask, 37)
    show_image((f_img[..., ::-1], 'automatic brightness and contrast'), dbg_ctx=dbg_ctx)

    # Noise removes
    blur = cv.pyrDown(f_img)
    blur = cv.pyrUp(blur)
    f_img = blur

    # CLAHE equalization
    lab = cv.cvtColor(f_img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=40.0, tileGridSize=(20, 20))
    clahe_img = clahe.apply(l)

    show_image([(l, 'luminosità'), (clahe_img, 'clahe')], dbg_ctx=dbg_ctx)

    _, l = cv.threshold(clahe_img, 50, 255, cv.THRESH_TOZERO)
    show_image(l, dbg_ctx=dbg_ctx)

    thr = cv.adaptiveThreshold(
        l, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 0)

    show_image((thr, 'thresholded'), dbg_ctx=dbg_ctx)

    return thr


def refine_region_above(img):
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
    morph = cv.morphologyEx(img, cv.MORPH_CLOSE, ker)
    return morph


def locate_seeds(img, dbg_ctx):
    h, w = img.shape[:2]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    show_image(hsv, cmap='magma', dbg_ctx=dbg_ctx)

    # res = cv.inRange(hsv, (15, 80, 100), (35, 255, 255))
    # res = cv.inRange(hsv, (15, 100, 100), (25, 255, 255))
    res = cv.inRange(hsv, (12, 100, 100), (25, 255, 255))
    show_image([res, img[..., ::-1]], dbg_ctx=dbg_ctx)

    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    morphed = cv.morphologyEx(res, cv.MORPH_OPEN, ker)

    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
    morphed = cv.morphologyEx(morphed, cv.MORPH_DILATE, ker)

    _, labels, stats, _ = cv.connectedComponentsWithStats(morphed)

    candidate_seeds_box = []
    yy, xx = np.mgrid[:h, :w]

    for i, stat in enumerate(stats[1:]):
        i = i + 1
        if stat[cv.CC_STAT_AREA] < 200:
            labels[i] = 0
        else:
            x, y = stat[cv.CC_STAT_LEFT], stat[cv.CC_STAT_TOP]
            width, height = stat[cv.CC_STAT_WIDTH], stat[cv.CC_STAT_HEIGHT]
            if width < 40 or height < 40:
                continue

            candidate_seeds_box.append((x, y, width, height))

            Logger().log(f'Seed {i} BB: ', Logger.LogLevel.LOCATE_SEEDS)
            Logger().log(f'\t x: {x}', Logger.LogLevel.LOCATE_SEEDS)
            Logger().log(f'\t y: {y}', Logger.LogLevel.LOCATE_SEEDS)
            Logger().log(f'\t w: {width}', Logger.LogLevel.LOCATE_SEEDS)
            Logger().log(f'\t h: {height}', Logger.LogLevel.LOCATE_SEEDS)

            cv.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)
            # cv.drawContours(img, [box], 0, (0, 0, 255), 1)
            # cx, cy = np.int0(center)
            # img[cy, cx] = [255, 0, 0]

    show_image(labels, cmap='magma', dbg_ctx=dbg_ctx)
    show_image(img[..., ::-1], dbg_ctx=dbg_ctx)

    return candidate_seeds_box
