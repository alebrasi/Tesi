import queue
from enum import Enum
from path_extraction.prune import valid_neighbors, seed_neighbors, root_neighbors, is_tip
import numpy as np
import math

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

class RootsExtraction:

    class PointType(Enum):
        TIP = 0,
        NODE = 1,
        INTERNAL = 2

    def __init__(self, skeleton, distances):
        self.skel = skeleton
        self.dist = skeleton * distances
        self.visited = set()

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
            n = valid_neighbors(cur_point, root_neighbors, self.skel)
            n.remove(prev_point)

            path.append(cur_point)

            # The point is a node
            if len(n) > 1:
                break

            if is_tip(cur_point, self.skel):
                res = self.PointType.TIP
                break

            prev_point = cur_point
            cur_point = n[0]

        return res, path

    def __extract_lstsq_from_path(self, path, max_residual_err, min_points_lstsq, iterations=-1):
        # FIXME: Scrivere meglio la documentazione
        # TODO: Testare meglio iterations
        """
        Returns a list of tuples containing the coefficients of the slopes and the path of points relative
        to that

        Parameters:
            path (list): List of points in (y, x) format
            max_residual_err (float): The maximum residual error for slope
            min_points_lstsq (int): Minimum points required to calculate the slope
            iterations (int): How many lines to calculate. 
                              Default=-1 calculate all the slopes in the path 
        """
        tmp_line_path = []
        line_vals = []
        slope_paths = []

        lines = []
        it = 0
        for i, p in enumerate(path, 1):

            if (it != -1) and (it == iterations):
                return lines

            tmp_line_path.append(p)

            if (len(tmp_line_path) > min_points_lstsq) or (i == len(path)):
                m, q, res = lstsq_from_path(tmp_line_path)

                """
                If the mean residual value is > max_residual_err, the latest point is removed
                and the current line values and relative set of points are added to the overall list
                """
                if len(res) > 0 and (res/len(path)) > max_residual_err:
                    last = tmp_line_path.pop()
                    m, q, _ = lstsq_from_path(tmp_line_path)
                    lines.append([(m, q), tmp_line_path])
                    tmp_line_path = [last]
                    it = it + 1

        # Check on the residual path
        if len(tmp_line_path) < min_points_lstsq:
            _, tmp = lines.pop()
            tmp_line_path = tmp + tmp_line_path

        m, q, _ = lstsq_from_path(tmp_line_path)
        lines.append([(m, q), tmp_line_path])

        return lines 

    def __compare_paths(self, source_path, neighbors_paths):
        """
        Returns the
        """
        # Grab the last least squared path
        mq, path = source_path[-1]
        print(mq)

        # For now the vector is created by taking the first and last node of the path
        vs = points_to_vector(path[0], path[-1])

        vn = []
        min_angle_vector = (361, None)

        # TODO: Try to make it with map() and min()
        for n in neighbors_paths:
            # Grab the first least squared path
            _, path = n[0]
            vn = points_to_vector(path[0], path[-1])
            angle = abs(angle_between_vectors(vs, vn))
            if angle < min_angle_vector[0]:
                min_angle_vector = (angle, n)

        smallest_angle_neighbor = min_angle_vector[1]

        return smallest_angle_neighbor 
        

    def extract_roots(self, s, max_residual_err=1.0, min_points_lstsq=4):
        """
        Returns the roots starting from the specified seed

        Parameters
        """

        neighbors = valid_neighbors(s, root_neighbors, self.skel)
        res = [ self.__walk_to_node(s, n) for n in neighbors ]

        # Retrieving the stem and forking node paths
        stem_paths = list(filter(lambda t: t[0] == self.PointType.TIP, res))
        forking_node_paths = list(filter(lambda t: t[0] == self.PointType.NODE, res))

        # Checks
        if len(stem_paths) == 0:
            raise Exception('Cannot find stem') 
        if len(forking_node_paths) == 0:
            raise Exception('Cannot find the forking node')
        if len(stem_paths) > 1:
            raise Exception('Found multiple stems')
        if len(forking_node_paths) > 1:
            raise Exception('Found multiple forking nodes')

        _, path = forking_node_paths[0]
        forking_node = path[-1]
        
        neighbors = valid_neighbors(forking_node, root_neighbors, self.skel)
        # Removes the father node
        neighbors.remove(path[-2])
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
            print(path1)
            a = self.__extract_lstsq_from_path(path1, max_residual_err, min_points_lstsq)
            res.append(a)
            
        path = self.__extract_lstsq_from_path(path, max_residual_err, min_points_lstsq)
        print("min: ", self.__compare_paths(path, res))
        
        stem_path = stem_paths[0]
        _, stem_path = stem_path
        return stem_path, res
        #"""