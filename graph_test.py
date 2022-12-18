import networkx as nx
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.graph import pixel_graph
from scipy.sparse import coo_matrix
import numpy as np
from utils import show_image
from skimage.morphology import skeletonize, medial_axis
import queue
import math
from path_extraction.prune import prune_skeleton_branches
from path_extraction.extract_roots import extract_plants_roots
from path_extraction.algorithm import RootsExtraction


def ravel_idx(x, y, width):
    return (y * width) + x

# Moore neighbor
def root_neighbors(p):
    y, x = p
    return [(y-1, x-1), (y-1, x), (y-1, x+1), (y, x-1), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]

def seed_neighbors(p):
    y, x = p
    return [(y, x-1), (y, x+1), (y+1, x-1), (y+1, x), (y+1, x+1)]

def valid_neighbors(p, neighbor_func, skel):
    return [ i for i in neighbor_func(p) if skel[i] ]

def is_tip(n, skel):
    return len(valid_neighbors(n, root_neighbors, skel)) == 1

def lstsq_from_path(path):
    """
    Fit a line, y = mx + q through the points in the array
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    """
    arr = np.array(path)
    y = arr[:, 0]
    x = arr[:, 1]

    A = np.vstack([x, np.ones(len(x))]).T
    sol, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    m, q = sol

    return m, q, residuals

def angle_between_two_slopes(m1, m2):
    return math.degrees(math.atan((m1 - m2) / 1 + (m1 * m2)))

def points_to_vector(p1, p2):
    """
    Returns a normalized vector given two points

    Parameters:
        p1: Source point
        p2: Destination point

    Returns:
        A numpy array containing the normalized vector
    """

    p1 = np.array(p1)
    p2 = np.array(p2)

    v = p1 - p2
    return v / np.linalg.norm(v)

def angle_between_vectors(v1, v2):
    """
    Returns the angle between two vectors normalized vectors
    https://www.euclideanspace.com/maths/algebra/vectors/angleBetween/

    Parameters:
        v1: First normalized vector
        v2: Second normalized vector
    """

    """
    Alternativa: 
    https://www.mathworks.com/matlabcentral/answers/180131-how-can-i-find-the-angle-between-two-vectors-including-directional-information
    a = np.cross(v1, v2)
    b = np.dot(v1, v2)
    print(math.degrees(np.arctan2(a, b)))
    """

    v1_x, v1_y = v1[:2]
    v2_x, v2_y = v2[:2]

    angle = math.atan2(v2_y, v2_x) - math.atan2(v1_y, v1_x)
    angle = math.degrees(angle)

    return angle

def boolean_matrix_to_rgb(m):
    m = m.copy().astype(np.uint8)
    m[m == 1] = 255
    return cv.cvtColor(m, cv.COLOR_GRAY2BGR)


def walk_to_first_available_node(n, neighbor_func=root_neighbors):
    path = []

    return path

skel = cv.imread('skel.png', cv.COLOR_BGR2GRAY)
# TODO: Controllare il comportamento di medial_axis
skel, distance = medial_axis(skel, return_distance=True)


skel1 = boolean_matrix_to_rgb(skel)

a = skel * distance

seeds = [ (140, 162), (130, 252), (132, 367) ]

# Skeleton branch prune
_, pruned = prune_skeleton_branches(seeds, skel.copy())

# Done just for retrieving the valid root tips.
# Can be swapped with a more efficient function
tips, pruned = prune_skeleton_branches(seeds, pruned) 

skel2 = boolean_matrix_to_rgb(pruned)

for tip in tips:
    skel2[tip] = (0, 255, 0)

show_image([skel2, skel1])

#extract_plants_roots(seeds, pruned)

"""
path = [(161, 170), (160, 169), (159, 169), (158, 169), (157, 168), (156, 168), (155, 168), 
        (155, 168), (154, 168), (153, 167), (152, 167), (151, 167), (150, 167)]

path1 = [(162, 172), (163, 173), (164, 174), (165, 175), (166, 176), (167, 176)]
path2 = [(163, 170), (164, 169), (165, 168), (166, 167), (167, 167)]

m, q, residuals = lstsq_from_path(path)
m1, q1, r1 = lstsq_from_path(path1)
m2, q2, r2 = lstsq_from_path(path2)

mean_error = residuals / len(path)
mean_error1 = r1 / len(path1)
mean_error2 = r2 / len(path2)

print(f'Angle between line and line1 {angle_between_two_slopes(m, m1)}')
print(f'Angle between line and line2 {angle_between_two_slopes(m, m2)}')

x1 = 20
x2 = 190

y1 = int(m * x1 + q)
y2 = int(m * x2 + q)

y1_1 = int(m1 * x1 + q1)
y2_1 = int(m1 * x2 + q1)

y1_2 = int(m2 * x1 + q2)
y2_2 = int(m2 * x2 + q2)


print(f'Mean error: {mean_error}')
print(f'Mean error 1: {mean_error1}')
print(f'Mean error 2: {mean_error2}')

cv.line(skel1, (x1, y1), (x2, y2), (0, 255, 0))
cv.line(skel1, (x1, y1_1), (x2, y2_1), (255, 0, 0))
#cv.line(skel1, (x1, y1_2), (x2, y2_2), (0, 0, 255))
show_image(skel1)

print(angle_between_vectors(v1, v2))
"""

s = seeds[0]
s = (157, 168)
alg = RootsExtraction(pruned, distance)
stem_path, paths = alg.extract_roots(s)
colors = [(255, 0, 0), (0,255,0), (0, 0, 255)]
print(len(paths))
skel1 = boolean_matrix_to_rgb(pruned)

for i, p in enumerate(stem_path):
    skel1[p] = colors[0]

show_image(skel1)

for path in paths:
    for i, p in enumerate(path):
        _, p = p
        #print(f'{i}): {p}')
        for point in p:
            skel1[point] = colors[i%len(colors)]


show_image(skel1)
#show_image(pruned * distance, cmap='magma')

"""
print(len(skel[skel != 0]))
G, nodes = pixel_graph(skel, connectivity=2)
print(nodes)
print(G)
print(np.argwhere(nodes == ravel_idx(165, 145, 500)))
"""




"""
print(ravel_idx(499, 499, 500))

pt1 = ravel_idx(165, 145, 500)
pt2 = ravel_idx(165, 91, 500)
print(G[pt1])
"""



