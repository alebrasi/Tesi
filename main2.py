import cv2 as cv
from skimage.morphology import skeletonize, medial_axis
from skimage.graph import pixel_graph
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import bm3d
import math

#from crf import CRF
from preprocess import adjust_gamma, automatic_brightness_and_contrast, clahe_bgr, remove_cc, locate_seed_line
from utils import find_file, show_image, f
from path_extraction.extract_roots import find_nearest

matplotlib.use('TKAgg')

image_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni'
mask_path = '/home/alebrasi/Documents/tesi/segmentate_prof/'
mask_extension = 'bmp'
image_extension = 'jpg'

# 88R, 89R SUS
# Fare test su 498 e 87R
# 105
image_name = '109'

mask_path = find_file(mask_path, f'{image_name}.{mask_extension}')
img_path = find_file(image_path, f'{image_name}.{image_extension}')

print(f'Image path: {img_path}')
print(f'Mask path: {mask_path}')

#img = cv.imread('/home/alebrasi/Documents/tesi/segmentate_prof/Sessione 1/Sessione 1.1/109.bmp')

img = cv.imread(img_path)

img, left_pt, right_pt = locate_seed_line(img, 5)

orig_img = img.copy()

h, w = img.shape[:2]
seed_line = np.zeros((h, w), dtype=np.uint8)
cv.line(seed_line, left_pt, right_pt, 255, 1)
show_image(seed_line)
show_image(img)

mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)
show_image(mask)

# ------------------- Mask refinement

# Morphological closing for filling small gaps in the mask
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker)

# Dilation from the seed line and below in order to account for segmentation errors
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask[left_pt[1]:, :] = cv.morphologyEx(
    mask[left_pt[1]:, :], cv.MORPH_DILATE, ker)
show_image(mask)

segmented = cv.bitwise_and(img, img, mask=mask)

f_img, alpha, beta = automatic_brightness_and_contrast(
    segmented, 10)  # TODO: Sperimentare sul clip limit
show_image(f_img)
clahe_img = clahe_bgr(f_img, 1, (30, 30))

show_image([f_img, clahe_img])

# Gamma adjustment
l = adjust_gamma(clahe_img, 1.5)
l1 = cv.equalizeHist(l)
show_image([l, l1])

# Removing noise due to gamma adjustment
_, l = cv.threshold(l, 25, 255, cv.THRESH_TOZERO)

#show_image([clahe_img, l])

thr = cv.adaptiveThreshold(
    l, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 0)
cc_rem = remove_cc(thr, 50)

ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
cc_rem[:left_pt[1], :] = cv.morphologyEx(
    cc_rem[:left_pt[1], :], cv.MORPH_CLOSE, ker)

# Blurring for smoothing the thresholded image
# TODO: Sostituibile con box filter?
blurred = cv.GaussianBlur(cc_rem, (3, 3), 5)

# Thresholding the blurred image in order to obtain a smoother segmentation
cc_rem = cv.adaptiveThreshold(
    blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 0)

cc_rem = remove_cc(cc_rem, 50)

# ------------------- Refined mask postprocess

one_px_ker = np.array([[1,  1, 1],
                       [1, -1, 1],
                       [1,  1, 1]])
diag_ker = []

diag_ker.append([[0, 1,  -1],
                 [0,  -1, 1],
                 [0,  0,  0]])

diag_ker.append([[-1, 1, 0],
                 [1,  -1, 0],
                 [0,  0, 0]])

diag_ker.append([[0,  0, 0],
                 [1,  -1, 0],
                 [-1, 1, 0]])

diag_ker.append([[0,  0,  0],
                 [0, -1,  1],
                 [0,  1, -1]])

rect_ker = []

rect_ker.append([[1,  1,  0],
                 [0, -1, -1],
                 [1,  1,  0]])

rect_ker.append([[0,   1, 1],
                 [-1, -1, 0],
                 [0,   1, 1]])

rect_ker.append([[0, -1,  0],
                 [1, -1,  1],
                 [1,  0,  1]])

rect_ker.append([[1, 0,  1],
                 [1, -1,  1],
                 [0,  -1,  0]])

bak = cc_rem.copy()

# Remove background pixel in a 1xn or nx1 shape surrounded by foreground pixel
for _ in range(4):
    for ker1 in rect_ker:
        xd = cv.morphologyEx(cc_rem, cv.MORPH_HITMISS, np.array(ker1))
        cc_rem = cc_rem + xd

# Remove baqckground pixel that has a neighbouring background pixel only diagonally
for ker in diag_ker:
    diag_px = cv.morphologyEx(cc_rem, cv.MORPH_HITMISS, np.array(ker))
    cc_rem = cc_rem + diag_px

# Remove background pixel surrounded by foreground pixel
one_px_gap = cv.morphologyEx(cc_rem, cv.MORPH_HITMISS, one_px_ker)
cc_rem = cc_rem + one_px_gap
show_image([bak, (cc_rem, "cc")])

# ------------------ Finding seeds

lx, ly = left_pt
rx, ry = right_pt

lx = 0 if lx < 0 else lx
rx = w if rx > w else rx

seed_roi = orig_img[:ly, lx:rx, ...]
hsv = cv.cvtColor(seed_roi, cv.COLOR_BGR2HSV)
res = cv.inRange(hsv, (15, 80, 100), (35, 255, 255))
show_image([res, seed_roi[..., ::-1]])

ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
ker2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
morphed = cv.morphologyEx(res, cv.MORPH_OPEN, ker)
# TODO: Dilate o close?
morphed = cv.morphologyEx(morphed, cv.MORPH_CLOSE, ker2)

_, labels, stats, _ = cv.connectedComponentsWithStats(morphed)

candidate_seeds_box = []

for i, stat in enumerate(stats[1:]):
    if stat[cv.CC_STAT_AREA] < 100:
        labels[i] = 0
    else:
        x, y = stat[cv.CC_STAT_LEFT], stat[cv.CC_STAT_TOP]
        width, height = stat[cv.CC_STAT_WIDTH], stat[cv.CC_STAT_HEIGHT]
        candidate_seeds_box.append((x, y, width, height))
        print(f'Seed {i} BB: ')
        print(f'\t x: {x}')
        print(f'\t y: {y}')
        print(f'\t w: {width}')
        print(f'\t h: {height}')

        cv.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 1)

show_image(labels, cmap='magma')
show_image(img)
# -------------------

cv.imwrite('res.png', cc_rem)
show_image([blurred, cc_rem])
# Mask normalization for skeletonize
cv.imwrite('skel.png', cc_rem)
cc_rem[cc_rem == 255] = 1

skeleton = skeletonize(cc_rem)
#from path_extraction.prune import prune_skeleton_branches
#_, pruned = prune_skeleton_branches(seeds, skel)

show_image([skeleton, medial_axis(cc_rem)])

skeleton = medial_axis(cc_rem).astype(np.uint8)
skeleton[skeleton == 1] = 255
skeleton_color = cv.cvtColor(skeleton, cv.COLOR_GRAY2BGR)

seeds_pos = []

for seed_bb in candidate_seeds_box:
    x, y, w, h = seed_bb

    """
    TODO: Migliorare il controllo. Con l'immagine 105 ci sono 3 intersezioni (bottom)
          Attualmente non tiene in conto in caso in cui il seme (scheletonizzato) intersechi solo il lato
          destro o sinistro della bb
    """
    ok = False
    while not ok:
        # Pixel intersection with each side of the rectangle bounding box
        top = skeleton[y, x:x+w]
        left = skeleton[y:y+h, x]
        right = skeleton[y:y+h, x+w]
        bottom = skeleton[y+h, x:x+w]

        n_top = cv.countNonZero(top)
        n_left = cv.countNonZero(left)
        n_right = cv.countNonZero(right)
        n_bottom = cv.countNonZero(bottom)

        # Decrease the size of the bounding box if any of the side of the rectangle intersect more than one point
        ok = True
        if n_bottom > 1:
            h = h-1
            ok = False
        if n_top > 1:
            y = y+1
            ok = False
        if n_left > 1:
            x = x+1
            ok = False
        if n_right > 1:
            w = w-1
            ok = False

    cv.rectangle(skeleton_color, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # Find, from the centroid, the nearest point on the stem
    centroid = (y+h//2, x+w//2)
    arr = skeleton[y:y+h, x:x+w]
    nodes = np.argwhere(arr) + [y, x]
    if len(nodes) > 0:
        nearest = find_nearest(centroid, nodes)
        a, b = nearest
        seeds_pos.append((a, b))
        skeleton_color[a, b] = (255,0,0)

    skeleton_color[y+h//2, x+w//2] = (0, 0, 255)
    bottom = cv.findNonZero(bottom)
    if bottom is not None:
        _, idx = bottom.flatten()
        #seeds_pos.append((y+h, x+idx))
        skeleton_color[y+h, x+idx] = (0, 0, 255)

print(seeds_pos)


show_image([skeleton_color, orig_img])
"""
res = cv.bitwise_and(img, img, mask=cc_rem)

show_image(res)

show_image([(l, 'Grayscale immagine rifinita'), (cc_rem, 'threshold'), (skeleton, 'skeleton maschera rifinita')])
"""
# Per il prune dello scheletro guardare nella cartella download 'Skeleton2Graph'
# https://ehu.eus/ccwintco/index.php?title=Skeletonization,_skeleton_pruning_and_simple_skeleton_graph_construction_example_in_Matlab
# http://www.ehu.es/ccwintco/uploads/a/a0/Skeleton2Graph.zip

# Codice originale
# https://cis.temple.edu/~latecki/Programs/BaiSkeletonPruningDCE.zip
"""
skel, distance = medial_axis(cc_rem, return_distance=True)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

show_image(dist_on_skel, cmap='magma')

dist = cv.distanceTransform(cc_rem, cv.DIST_L2, cv.DIST_MASK_PRECISE)

show_image(dist)
"""
