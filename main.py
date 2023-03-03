import cv2 as cv
from skimage.morphology import skeletonize, medial_axis, thin
from skimage.graph import pixel_graph
from skimage.exposure import match_histograms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import argparse
import sys
import random
import json

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

def get_measures(plant_id, plant, distances):
    # Longest root

    measures = {}

    measures['plant_id'] = plant_id
    roots = plant.roots
    longest_root = max(roots, key=lambda r: len(r.points))
    print('Max root lenght: ', len(root.points))
    measures['max_root_lenght'] = len(root.points)

    # Num roots
    print('Number of roots:', len(plant.roots))
    measures['roots_number'] = len(plant.roots)

    # Stem angle
    print('Stem angle: ', plant.stem.angle)
    measures['stem_angle'] = round(plant.stem.angle, 2)

    # Root thickness
    roots_thickness = []
    all_points = set()
    measures['roots_thickness'] = {}
    for i, r in enumerate(roots):
        points = r.points
        tmp = list(map(lambda p: distances[p], points))
        mean_thickness = sum(tmp)/len(tmp)
        measures['roots_thickness'][i+1] = round(mean_thickness, 2)

        print(f'Root {i+1}  mean thickness: ', mean_thickness)
        for i, point in enumerate(points):
            roots_thickness.append(tmp[i])
            y, x = point
            all_points.add((x, y))

    avg_roots_thickness = sum(roots_thickness)/len(roots_thickness)
    print('Average roots thickness: ', avg_roots_thickness)
    measures['average_roots_thickness'] = round(avg_roots_thickness, 2)

    # Convex hull
    hull = cv.convexHull(np.array(list(all_points)))
    hull_area = cv.contourArea(hull)
    print('Plant convex hull: ', hull_area)
    measures['convex_hull'] = round(hull_area, 2)

    return measures


matplotlib.use('TKAgg')
cv.setRNGSeed(123)
random.seed(123)

image_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/resize'
mask_path = '/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/segmentate_1024'

mask_extension = 'png'
image_extension = 'jpg'
invert_mask = True

image_name = '1004'
reference_hist_img_name = '995'

parser = argparse.ArgumentParser()
parser.add_argument('image_name', metavar='IMG')
parser.add_argument('--no_extraction', action='store_true')
parser.add_argument('--d_seed_line', action='store_true', default=False)
parser.add_argument('--d_post_process', action='store_true', default=False)

d_seed_line = False
d_post_process = False
no_extraction = False
d_region_below = False
d_region_above = False
d_locate_seeds = False

#"""
args = parser.parse_args()
no_extraction = args.no_extraction
image_name = args.image_name
d_seed_line = args.d_seed_line
d_post_process = args.d_post_process
#"""

dbg_ctx_seed_line = DebugContext('seed_line', d_seed_line)
dbg_ctx_post = DebugContext('post_process', d_post_process)
dbg_ctx_region_below = DebugContext('region_below', d_region_below)
dgb_ctx_region_above = DebugContext('region_abow', d_region_above)
dbg_ctx_locate_seeds = DebugContext('locate_seeds', d_locate_seeds)

mask_path = find_file(mask_path, f'{image_name}.{mask_extension}')
img_path = find_file(image_path, f'{image_name}.{image_extension}')
reference_hist_path = find_file(image_path, f'{reference_hist_img_name}.{image_extension}')

print(f'Image path: {img_path}')
print(f'Mask path: {mask_path}')

#img = cv.imread('/home/alebrasi/Documents/tesi/segmentate_prof/Sessione 1/Sessione 1.1/109.bmp')

img = cv.imread(img_path)
mask = cv.imread(mask_path, cv.COLOR_BGR2GRAY)
reference_hist_image = cv.imread(reference_hist_path)


show_image(mask)
if invert_mask:
    mask = cv.bitwise_not(mask)
    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
    show_image(mask, dbg_ctx=dbg_ctx_post)

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

matched_hist_img = match_histograms(img, reference_hist_image, channel_axis=-1)
show_image([img[..., ::-1], matched_hist_img[..., ::-1]], dbg_ctx=dbg_ctx_post)
segmented = cv.bitwise_and(matched_hist_img, matched_hist_img, mask=mask)
show_image((segmented[..., ::-1], 'segmentata'), dbg_ctx=dbg_ctx_post)

region_below = segmented[left_pt[1]:, :]
region_above = segmented[:left_pt[1], :]
mask_above = mask[:left_pt[1], :]
mask_below = mask[left_pt[1]:, :]
show_image(region_below[..., ::-1])
# ------------ Seeds localization ----------------------------------
top_left_pt, bottom_right_pt = seed_line_roi[0], seed_line_roi[1]
br_y, br_x = bottom_right_pt
seeds_roi = img[:br_y, :br_x]

seeds_bb = locate_seeds(seeds_roi.copy(), dbg_ctx_locate_seeds)
# -----------------------------------------------------------------

# -------------- Mask refinement ------------------
h, w = segmented.shape[:2]
refined_mask = np.zeros((h, w), dtype=np.uint8)

refined_mask[left_pt[1]:, :] = refine_region_below(region_below, mask_below, dbg_ctx=dbg_ctx_region_below)
refined_mask[:left_pt[1], :] = refine_region_above(mask_above, dgb_ctx_region_above)

# Morphological closing inside seeds bounding box
ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
for bb in seeds_bb:
    x, y, w, h = bb
    tmp = refined_mask[y:y+h, x:x+w]
    refined_mask[y:y+h, x:x+w] = cv.morphologyEx(tmp, cv.MORPH_CLOSE, ker)
show_image(refined_mask, dbg_ctx=dbg_ctx_post)
smoothed_mask = cv.medianBlur(refined_mask, 5)       # Edge smoothing

#ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
#asd = cv.morphologyEx(asd, cv.MORPH_ERODE, ker)

# Filling gaps with area < 30 px
cnts, _ = cv.findContours(smoothed_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
for i, cnt in enumerate(cnts):
    if cv.contourArea(cnt) < 30:
        cv.drawContours(smoothed_mask, cnts, i, 255, -1)

show_image([(refined_mask, 'maschera rifinita'), (smoothed_mask, 'bordi smussati')], dbg_ctx=dbg_ctx_post)
show_image((cv.bitwise_and(orig_img, orig_img, mask=smoothed_mask)[..., ::-1], 'risultato segmentazione'), dbg_ctx=dbg_ctx_post)
show_image((smoothed_mask, "bordi smussati"), dbg_ctx=dbg_ctx_post)
refined_mask = smoothed_mask
# -------------------------------------------------------------------------------

# ----------------- Skeletonization and prune -----------------------------------

skeleton, dist = medial_axis(refined_mask.astype(bool), return_distance=True, random_state=123)
show_image(skeleton, dbg_ctx=dbg_ctx_post)
skeleton = skeleton.astype(np.uint8)
cross_ker = np.array([[-1, 1, -1],
                      [1, -1, 1],
                      [-1, 1, -1]], dtype=np.uint8)
cross_nodes = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, cross_ker)
skeleton = skeleton | cross_nodes

pruned_skeleton = prune3(skeleton, 10)
pruned_skeleton = thin(pruned_skeleton.astype(bool)).astype(np.uint8)

show_image([(skeleton, 'scheletro'), (pruned_skeleton, 'scheletro con branch <= 10px potati')], dbg_ctx=dbg_ctx_post)

pruned_skeleton = pruned_skeleton.astype(bool)
# -------------------------------------------------------------------------------

tmp = pruned_skeleton.copy().astype(np.uint8)
tmp = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
tmp[pruned_skeleton, ...] = [255, 255, 255]
seeds_pos = []
for bb in seeds_bb:
    x, y, w, h = bb

    centroid = (y+h//2, x+w//2)
    arr = pruned_skeleton[y:y+h, x:x+w]
    nodes = np.argwhere(arr) + [y, x]
    if len(nodes) > 0:
        nearest = find_nearest(centroid, nodes)
        cv.rectangle(tmp, bb, (0,255,0), 1)
        a, b = nearest
        tmp[centroid[0], centroid[1], ...] = [0,255,255]
        tmp[a, b, ...] = [0, 0,255]
        seeds_pos.append((a, b))
show_image(tmp[:, ::-1, ::-1], dbg_ctx=dbg_ctx_post)
print(seeds_pos)
show_image(pruned_skeleton, dbg_ctx=dbg_ctx_post)

plants = extraction(seeds_pos, pruned_skeleton, dist, orig_img)

spline_parameters = {
                        'tension': 0.5,
                        'spacing': 50 
                    }

# Create Plant structure
rootnav_plants = [ RootNavPlant(idx, 'orzo', p.stem, p.seed_coords) for idx, p in enumerate(plants) ]

measures = {}
measures['plants'] = []

for i, plant in enumerate(plants):
    for root in plant.roots:
        points = root.points
        diameters = [ dist[p] for p in points ]
        points = [ (x, y) for y, x in points ]
        tmp_root = RootNavRoot(points, 
                                diameters, 
                                spline_tension=spline_parameters['tension'], 
                                spline_knot_spacing=spline_parameters['spacing']
                              )
        rootnav_plants[i].roots.append(tmp_root)
    measures['plants'].append(get_measures(i+1, plant, dist))

output_dir = '/home/alebrasi/Videos/rsml_tests'

# Dumps measures to json
with open(f'{output_dir}/{image_name}.json', 'w', encoding='utf-8') as f:
    json_measures = json.dumps(measures, ensure_ascii=False, indent=4)
    f.write(json_measures)

# Output to RSML
RSMLWriter.save(image_name, output_dir, rootnav_plants)