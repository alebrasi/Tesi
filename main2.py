import cv2 as cv
from skimage.morphology import skeletonize, medial_axis
from skimage.graph import pixel_graph
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bm3d
import math

from crf import CRF
from preprocess import adjust_gamma, automatic_brightness_and_contrast, clahe_bgr, remove_cc, locate_seed_line
from utils import find_file, show_image, f

matplotlib.use('TKAgg')

image_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni'
mask_path = '/home/alebrasi/Documents/tesi/segmentate_prof/'
mask_extension = 'bmp'
image_extension = 'jpg'

# 88R, 89R SUS
image_name = '109'

mask_path = find_file(mask_path, f'{image_name}.{mask_extension}')
img_path = find_file(image_path, f'{image_name}.{image_extension}')

print(f'Image path: {img_path}')
print(f'Mask path: {mask_path}')

#img = cv.imread('/home/alebrasi/Documents/tesi/segmentate_prof/Sessione 1/Sessione 1.1/109.bmp')

img = cv.imread(img_path)

# FIXME: In locate seed line aggiungere inRange per individuare solo le righe nere
img, left_pt, right_pt = locate_seed_line(img)

h, w = img.shape[:2]
seed_line = np.zeros((h, w), dtype=np.uint8)
cv.line(seed_line, left_pt, right_pt, 255, 1)
show_image(img)


mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)
show_image(mask)

# Morphological closing for filling small gaps in the mask
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker)

segmented = cv.bitwise_and(img, img, mask=mask)

f_img, alpha, beta = automatic_brightness_and_contrast(segmented, 50) # TODO: Sperimentare sul clip limit

clahe_img = clahe_bgr(f_img, 1, (30, 30)) 

# Gamma adjustment
l  = adjust_gamma(clahe_img, 1.5)

# Removing noise due to gamma adjustment
_, l = cv.threshold(l, 25, 255, cv.THRESH_TOZERO)

show_image(l)

#show_image([clahe_img, l])

thr = cv.adaptiveThreshold(l, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 0)
cc_rem = remove_cc(thr, 50)

show_image(cc_rem)

ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
cc_rem[:left_pt[1], :] = cv.morphologyEx(cc_rem[:left_pt[1], :], cv.MORPH_CLOSE, ker)

# Blurring for smoothing the thresholded image
blurred = cv.GaussianBlur(cc_rem, (3,3), 5) # TODO: Sostituibile con box filter?

show_image(blurred)

# Thresholding the blurred image in order to obtain a smoother segmentation
cc_rem = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 0)

# Removing diagonal pixels and one background px surrounded by foreground
# TODO: Aggiungere altri filtri
ker = np.array([[0, -1, 1],
                [1, 1, -1],
                [1, 1, 0]])

ker2 = np.array([[1, 1, 1],
                 [1, -1, 1],
                 [1, 1, 1]])

one_px_diag = cv.morphologyEx(cc_rem, cv.MORPH_HITMISS, ker)
one_px_gap = cv.morphologyEx(cc_rem, cv.MORPH_HITMISS, ker2)

cc_rem = (cc_rem + one_px_gap) - one_px_diag
show_image(cc_rem)

# ------------------ Finding seeds
# TODO: Gestire il caso in cui le radici intersecano la seed line
candidate_seeds = cv.bitwise_and(cc_rem, cc_rem, mask=seed_line)
_, _, stats, centroids = cv.connectedComponentsWithStats(candidate_seeds)
tmp = img.copy()
seeds = []
for centroid in np.uint16(centroids)[1:]:
    tmp[centroid[1], centroid[0]] = (255, 0, 0)
    seeds.append((centroid[1], centroid[0]))
show_image((tmp, 'Seeds location'))
#-------------------

cv.imwrite('res.png', cc_rem)
show_image([blurred, cc_rem])
# Mask normalization for skeletonize
cv.imwrite('skel.png', cc_rem)
cc_rem[cc_rem == 255] = 1

skeleton = skeletonize(cc_rem)

graph, nodes = pixel_graph(skeleton, connectivity=8)
print(graph.shape)

for node in nodes[:10]:
    print(np.unravel_index(node, (500, 500)))

print(graph)
show_image(skeleton)


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