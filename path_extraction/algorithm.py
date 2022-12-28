import queue
from enum import Enum
from path_extraction.prune import valid_neighbors, seed_neighbors, root_neighbors, is_tip
import numpy as np
import math
from path_extraction.path import Path
from collections import namedtuple

class RootsExtraction:

    class PointType(Enum):
        TIP = 0,
        NODE = 1,
        INTERNAL = 2

    def __init__(self, skeleton, distances):
        self._skel = skeleton
        self._dist = skeleton * distances
        self._visited = set()
        self._WalkedPath = namedtuple('WalkedPath', ['point_type', 'path'])

    def __walk_to_node(self, sender_p, start_p):
        # TODO: Cambiare il nome alla funzione
        """
        Returns a list of points (path) that are between the point `p` and the first available node or tip

        Paramteres:
            sender_p (tuple): The `father` of the starting point
            start_p (tuple): The starting point, which is the `son` (or neighbor) of `sender_p` 

        Returns:
            path: A list containing the points walked
        """

        prev_point = sender_p
        cur_point = start_p
        path = []
        res = self.PointType.NODE

        while True:
            n = valid_neighbors(cur_point, root_neighbors, self._skel)
            n.remove(prev_point)

            path.append(cur_point)

            # The point is a node
            if len(n) > 1:
                break

            if is_tip(cur_point, self._skel):
                res = self.PointType.TIP
                break

            prev_point = cur_point
            cur_point = n[0]

        return self._WalkedPath(res, Path(path))

    def extract_roots(self, s, max_residual_err=1.0, min_points_lstsq=4):
        """
        Returns the roots starting from the specified seed

        Parameters
        """

        neighbors = valid_neighbors(s, root_neighbors, self._skel)
        res = [ self.__walk_to_node(s, n) for n in neighbors ]

        # Retrieving the stem and forking node paths
        stem_paths = list(filter(lambda t: t.point_type == self.PointType.TIP, res))
        forking_node_paths = list(filter(lambda t: t.point_type == self.PointType.NODE, res))
        print(forking_node_paths)

        # Checks
        """
        if len(stem_paths) == 0:
            raise Exception('Cannot find stem') 
        if len(forking_node_paths) == 0:
            raise Exception('Cannot find the forking node')
        if len(stem_paths) > 1:
            raise Exception('Found multiple stems')
        if len(forking_node_paths) > 1:
            raise Exception('Found multiple forking nodes')
        """
        _, path = forking_node_paths[0]
        forking_node = path.points[-1]
        
        neighbors = valid_neighbors(forking_node, root_neighbors, self._skel)
        # Removes the father node
        print(neighbors)
        print(path.points)
        neighbors.remove(path.points[-2])
        """
        root_paths = [path]

        while True:
            neighbors = valid_neighbors(forking_node, root_neighbors, self.skel)
            # Removes the father node
            neighbors.remove(path[-2])
            tmp = []
            for n in neighbors:
                px_type, path = self.__walk_to_node(forking_node, n)
                tmp.append(self.__extract_lstsq_from_path(path, max_residual_err, min_points_lstsq))
            break
        """

        
        #"""
        res = []
        for n in neighbors:
            _, path1 = self.__walk_to_node(forking_node, n)
            a = path1.get_lstsq_paths(max_residual_err, min_points_lstsq, iterations=-1)
            res.append(a)
            
        path = path.get_lstsq_paths(max_residual_err, min_points_lstsq)[-1]

        t = min(map(lambda p: p[0], res), key=lambda k: path.angle_between(k))
        print(t)
        stem_path = stem_paths[0]
        _, stem_path = stem_path
        return stem_path, res
        #"""