import networkx as nx
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.graph import pixel_graph
import numpy as np
from utils import show_image
from skimage.morphology import skeletonize, medial_axis
import queue
import math
import time

from path_extraction.prune import prune_skeleton_branches
from path_extraction.algorithm import RootsExtraction

from graph.graph_creation import create_graph, PointType
from graph.graph_drawing import draw_graph
from graph.algorithm import extract
from graph.plant import Plant
from utils import find_file


def boolean_matrix_to_rgb(m):
    m = m.copy().astype(np.uint8)
    m[m == 1] = 255
    return cv.cvtColor(m, cv.COLOR_GRAY2BGR)

def hsv2bgr(h, s=255, v=255):
    c = np.uint8([[[h, s, v]]])
    return cv.cvtColor(c, cv.COLOR_HSV2BGR).reshape(1,1,3)[0][0]

def extraction(seeds, skel, distance, orig_img):
    skel2 = boolean_matrix_to_rgb(skel)
    h, w = skel.shape[:2]

    G = create_graph(seeds, skel, distance)
    color_map = { 
                    PointType.NODE:'lightgreen', 
                    PointType.SOURCE:'red', 
                    PointType.TIP:'orange' 
                }

    node_color = [ color_map[G.nodes[node]['node_type']] for node in G ]

    print(orig_img.shape)
    draw_graph(G, with_labels=True, node_color=node_color, node_size=20, invert_xaxis=True)
    plants = extract(G)

    all_p = orig_img
    all_p[skel, ...] = [0,0,0]
    show_image(all_p[..., ::-1])
    cv.namedWindow('image', cv.WINDOW_FULLSCREEN)
    plants1=[]
    colors = [(238, 255, 150), (0, 0, 255), (239, 0, 255)]
    #time.sleep(10)
    for i, plant in enumerate(plants, 0):
        print(f'Plant: {i}')
        print(f'Num roots: {len(plant.roots)}')
        mask = np.zeros((h, w, 3))
        #color1 = (list(np.random.choice(range(150, 255), size=3)))
        #hue = None
        #while (hue == None) or (hue > 35 and hue < 86) or (hue > 105 and hue < 135):
        #    hue = np.random.choice(range(1, 179), size=1)
        #color1 = hsv2bgr(hue)
        for root in plant.roots:
            print(root.edges)
            print(root._edges)
            print(root._split_node)
            print('\n\n\n')
            #color1 = (list(np.random.choice(range(150, 255), size=3)))  
            color1 = colors[i]
            points = np.array(root.points)
            for point in points:
                #print(point)
                y, x = point
                
                mask[y, x, i] = 255
                all_p[y, x, :] = color1
                for y1 in range(y-1, y+2):
                    for x1 in range(x-1, x+2):
                #all_p[y-1, x-1, :] = color1
                #all_p[y+1, x+1, :] = color1
                        all_p[y1, x1, :] = color1
                cv.imshow('image', all_p.astype(np.uint8))
                cv.waitKey(1)
                #time.sleep(0.01)
        
        print('Stem:')
        points = np.array(plant.stem.points)
        print(plant.stem.edges)
        for point in points:
            y, x = point
            for y1 in range(y-1, y+2):
                for x1 in range(x-1, x+2):
                    all_p[y1, x1, :] = (0, 255 , 0)
            cv.imshow('image', all_p.astype(np.uint8))
            cv.waitKey(1)
        #time.sleep(10)
    print('Done!')
    
    cv.imshow('image', all_p.astype(np.uint8))
    while True:
        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()

    #draw_graph(G, with_labels=True, node_color=node_color, node_size=20, invert_xaxis=False)
    return plants

if __name__ == '__main__':
    dataset_path = '/home/alebrasi/Documents/tesi/Dataset'
    img_name = '88R'

    #seeds = [(146, 165), (149, 255), (151, 345)]
    #seeds = [(164, 172), (166, 255), (166, 340)]
    #seeds = [(147, 333), (150, 145), (156, 241)]
    seeds = [(138, 319), (141, 136), (140, 229)]

    img_path = find_file(dataset_path, img_name)
    img = cv.imread(img_path)

    skel = cv.imread('skel.png', cv.COLOR_BGR2GRAY)
    # TODO: Controllare il comportamento di medial_axis
    skel, distance = medial_axis(skel, return_distance=True)


    a = skel * distance

    _, pruned = prune_skeleton_branches(seeds, skel)
    extraction(seeds, pruned, distances, img)


