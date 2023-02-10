import cv2 as cv
from skimage.morphology import skeletonize, medial_axis, thin
from skimage.graph import pixel_graph
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import argparse
import sys

from preprocess import adjust_gamma, automatic_brightness_and_contrast, clahe_bgr, remove_cc, locate_seed_line
from utils import find_file, show_image, f, DebugContext
from path_extraction.prune import prune_skeleton_branches, prune3
from graph_test import extraction
from refine_mask import refine_region_above, refine_region_below, locate_seeds

from graph.plant import Plant, Root
from rsml.rsmlwriter import RSMLWriter
from rsml.plants import Plant as RootNavPlant, Root as RootNavRoot

def find_nearest(node, nodes):
    node = np.array(node)
    nodes = np.array(nodes)
    d = np.linalg.norm(node-nodes, axis = 1)
    p = np.argmin(d)

    return nodes[p]

matplotlib.use('TKAgg')

#image_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni'
#mask_path = '/home/alebrasi/Documents/tesi/segmentate_prof/'

image_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/resize'
mask_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/segmentate_1024'

mask_extension = 'png'
image_extension = 'jpg'
invert_mask = True

# 88R, 89R SUS
# Fare test su 498 e 87R
# 105, 950R
image_name = '109R'


parser = argparse.ArgumentParser()
parser.add_argument('image_name', metavar='IMG')
parser.add_argument('--no_extraction', action='store_true')
parser.add_argument('--d_seed_line', action='store_true', default=False)
parser.add_argument('--d_post_process', action='store_true', default=False)

d_seed_line = False
d_post_process = False
no_extraction = False

#"""
args = parser.parse_args()
no_extraction = args.no_extraction
image_name = args.image_name
d_seed_line = args.d_seed_line
d_post_process = args.d_post_process
#"""
d_region_below = True
d_region_above = True
d_locate_seeds = True

dbg_ctx_seed_line = DebugContext('seed_line', d_seed_line)
dbg_ctx_post = DebugContext('post_process', d_post_process)
dbg_ctx_region_below = DebugContext('region_below', d_region_below)
dgb_ctx_region_above = DebugContext('region_abow', d_region_above)
dbg_ctx_locate_seeds = DebugContext('locate_seeds', d_locate_seeds)

mask_path = find_file(mask_path, f'{image_name}.{mask_extension}')
img_path = find_file(image_path, f'{image_name}.{image_extension}')

print(f'Image path: {img_path}')
print(f'Mask path: {mask_path}')

#img = cv.imread('/home/alebrasi/Documents/tesi/segmentate_prof/Sessione 1/Sessione 1.1/109.bmp')

img = cv.imread(img_path)
mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)

if invert_mask:
    mask = cv.bitwise_not(mask)
    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

h, w = img.shape[:2]

seed_line_roi = [[148, 0], [300, w-1]]

M, left_pt, right_pt = locate_seed_line(img, seed_line_roi, 5, dbg_ctx=dbg_ctx_seed_line)

# Alignment in order to subdivide the above and below regions with even height
# Must be done for image pyramids upscale
if ((h - left_pt[1]) % 2) == 1:
    y = left_pt[1] + 1
    x1 = left_pt[0]
    x2 = right_pt[0]

    left_pt = (x1, y)
    right_pt = (x2, y)

# Apply rotation matrix to image and mask
img = cv.warpAffine(img, M, (w, h))
mask = cv.warpAffine(mask, M, (w, h))

orig_img = img.copy()

tmp = img.copy()
cv.line(tmp, left_pt, right_pt, (0, 255, 0), 3)
show_image(tmp[..., ::-1], dbg_ctx=dbg_ctx_seed_line)
show_image(img[..., ::-1], dbg_ctx=dbg_ctx_seed_line)

show_image(mask, dbg_ctx=dbg_ctx_seed_line)

# ------------------- Mask preparation ------------------------------

# Morphological closing for filling small gaps in the mask
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker)

show_image((mask, 'maschera'), dbg_ctx=dbg_ctx_post)

segmented = cv.bitwise_and(img, img, mask=mask)

show_image((segmented, 'segmentata'), dbg_ctx=dbg_ctx_post)

region_below = segmented[left_pt[1]:, :]
region_above = segmented[:left_pt[1], :]
mask_above = mask[:left_pt[1], :]

# ------------ Seeds localization ----------------------------------
top_left_pt, bottom_right_pt = seed_line_roi[0], seed_line_roi[1]
br_y, br_x = bottom_right_pt
seeds_roi = segmented[:br_y, :br_x]

seeds_bb = locate_seeds(seeds_roi.copy(), dbg_ctx_locate_seeds)
# -----------------------------------------------------------------

# -------------- Mask refinement ------------------
h, w = segmented.shape[:2]
refined_mask = np.zeros((h, w), dtype=np.uint8)

refined_mask[left_pt[1]:, :] = refine_region_below(region_below, dbg_ctx=dbg_ctx_region_below)
refined_mask[:left_pt[1], :] = refine_region_above(mask_above, dgb_ctx_region_above)

# Morphological closing inside seeds bounding box
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
for bb in seeds_bb:
    x, y, w, h = bb
    tmp = refined_mask[y:y+h, x:x+w]
    refined_mask[y:y+h, x:x+w] = cv.morphologyEx(tmp, cv.MORPH_CLOSE, ker)

smoothed_mask = cv.medianBlur(refined_mask, 5)       # Edge smoothing

#ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
#asd = cv.morphologyEx(asd, cv.MORPH_ERODE, ker)

# Filling gaps with area < 30 px
cnts, _ = cv.findContours(smoothed_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
for i, cnt in enumerate(cnts):
    if cv.contourArea(cnt) < 30:
        cv.drawContours(smoothed_mask, cnts, i, 255, -1)

show_image([(refined_mask, 'maschera rifinita'), (smoothed_mask, 'bordi smussati')])
show_image((cv.bitwise_and(orig_img, orig_img, mask=smoothed_mask)[..., ::-1], 'risultato segmentazione'))

refined_mask = smoothed_mask
# -------------------------------------------------------------------------------

# ----------------- Skeletonization and prune -----------------------------------

skeleton, dist = medial_axis(refined_mask.astype(bool), return_distance=True)
pruned_skeleton = prune3(skeleton.astype(np.uint8), 6)

show_image([(skeleton, 'scheletro'), (pruned_skeleton, 'scheletro con branch < 6px potati')])

pruned_skeleton = pruned_skeleton.astype(bool)
# -------------------------------------------------------------------------------

seeds_pos = []
for bb in seeds_bb:
    x, y, w, h = bb

    centroid = (y+h//2, x+w//2)
    arr = pruned_skeleton[y:y+h, x:x+w]
    nodes = np.argwhere(arr) + [y, x]
    if len(nodes) > 0:
        nearest = find_nearest(centroid, nodes)
        a, b = nearest
        seeds_pos.append((a, b))

print(seeds_pos)
show_image(pruned_skeleton)

plants = extraction(seeds_pos, pruned_skeleton, dist, orig_img)

spline_parameters = {
                        'tension': 0.5,
                        'spacing': 20 
                    }

# Create Plant structure
rootnav_plants = [ RootNavPlant(idx, 'orzo') for idx in range(len(plants)) ]

for i, plant in enumerate(plants):
    for root in plant.roots:
        points = root.points
        points = [ (x, y) for y, x in root.points ]
        tmp_root = RootNavRoot(points, spline_tension=spline_parameters['tension'], spline_knot_spacing=spline_parameters['spacing'])
        rootnav_plants[i].roots.append(tmp_root)

output_dir = '/home/alebrasi/Videos/rsml_tests'

# Output to RSML
RSMLWriter.save(image_name, output_dir, rootnav_plants)