import math

from graph.create import PointType


def flat_map(f, xs): return (y for ys in xs for y in f(ys))


def mirror_angle_y_axis(angle):
    rad = math.radians(angle)
    c = math.cos(rad)
    s = math.sin(rad)
    a = math.atan2(s, -c)
    a = math.degrees(a)
    return a + 360 * (1 if a < 0 else 0)


def calc_ema(ema, angle, alpha):
    return (alpha * angle) + ((1 - alpha) * ema)


def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class Root:
    G = None

    @staticmethod
    def attach_graph(G):
        Root.G = G

    def __init__(self, plant, start_edge, ema_alpha=0.20):
        self._edges = set()
        self._cur_edge = None
        self._angle_ema = None
        self._ordered_edges = []
        self._ema_alpha = ema_alpha
        self._start_edge = start_edge
        self._plant = plant
        self._add_edge(start_edge)

        self._split_node = None

    @property
    def angle(self):
        return mirror_angle_y_axis(self._angle_ema)

    def __str__(self):
        return f'Start edge: {self._start_edge} \n \
                Cur edge: {self._cur_edge} \n \
                Seed: {self._plant._seed} \n \
                Stem: {self._plant._tip_start_node} \n \
                Edges: {self._ordered_edges} \n'

    def _add_edge(self, edge):
        """
        Add an edge to the root

        Parameters:
            - edge (tuple) : tuple that describes the edge
        """
        G = type(self).G
        G.edges[edge]['walked'] = True
        self._update_angle_EMA(edge)
        self._edges.add(edge)
        self._ordered_edges.append(edge)
        self._cur_edge = edge

    def non_walked_neighbors(self, node):
        """
        Returns the node neighbors that have not been walked from the given node.
        A walk is present if the current root contains an edge between the node and
        its neighbor.

        Parameters:
            - node (int) : the node to check

        Returns:
        list of integers of the not walked node neighbors
        """

        G = type(self).G

        neighbors = []

        for n in G.neighbors(node):
            if not (G.edges[(node, n)]['walked'] or G.edges[(n, node)]['walked']):
                neighbors.append(n)

        return neighbors

    @property
    def highest_non_walked_node(self):
        G = type(self).G
        non_walked_nodes = []

        for node in self.nodes:
            neighbors = self.non_walked_neighbors(node)
            for neighbor in neighbors:
                if neighbor == self._plant._stem_start_node:
                    continue
                non_walked_nodes.append((node, neighbor, G.nodes[node]['pos']))

        return min(non_walked_nodes,
                   key=lambda n: n[2][0],  # key = y coordinate of 'pos'
                   default=None)  # [0]

    def copy_until_node(self, node, node_neighbor):
        tmp_root = Root(self._plant, self._start_edge)
        tmp_root._split_node = (node, node_neighbor)
        tmp_edges = [True if edge[1] == node else False for edge in self._ordered_edges]
        idx = tmp_edges.index(True)
        edges = self._ordered_edges[1:idx + 1]  # Skips the first edge, which is added when the root is initialized
        for edge in edges:
            tmp_root._add_edge(edge)
        tmp_root._add_edge((node, node_neighbor))
        return tmp_root

    @property
    def nodes(self):
        tmp = set()
        for e in self._edges:
            tmp.add(e[1])
        return tmp

    @property
    def nodes_coords(self):
        G = type(self).G
        tmp = []
        for e in self._ordered_edges:
            pos = G.nodes[e[0]]['pos']
            tmp.append(pos)

        last_edge = self._ordered_edges[-1]
        pos = G.nodes[last_edge[1]]['pos']
        tmp.append(pos)

        return tmp

    @property
    def edges(self):
        """
        Returns all the edges in the root
        """
        return self._ordered_edges.copy()

    @property
    def points(self):
        G = type(self).G
        return list(flat_map(lambda p: p, (G.edges[e]['path_points']
                                           for e in self._ordered_edges)))

    @property
    def cur_edge(self):
        return self._cur_edge

    def _update_angle_EMA(self, edge):
        """
        Calculate the exponential moving average of the angle
        https://www.investopedia.com/terms/e/ema.asp
        https://towardsdatascience.com/moving-averages-in-python-16170e20f6c

        Versione utilizzata:
        https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/
        """

        G = type(self).G
        cur_angle = G.edges[edge]['weight']

        edge_len = G.edges[edge]['length']
        ema_alpha = 0.8
        if edge_len < 25:
            ema_alpha = map_range(edge_len, 1, 25, 0.01, 0.8)

        if self._angle_ema is None:
            self._angle_ema = cur_angle
            return cur_angle

        cur_angle_ema = calc_ema(self._angle_ema, cur_angle, ema_alpha)
        self._angle_ema = cur_angle_ema

    def _compare_neighbor(self, node, neighbor):
        G = type(self).G
        if neighbor == self._plant._stem_start_node:
            return float('inf')

        edge = (node, neighbor)
        return math.fabs(self._angle_ema - G.edges[edge]['weight'])

    def _node_neighbors(self, edge):
        """
        Returns the neighbors of the destination node given the directed edge.
        E.g. (0, 1) describes the edge from node 0 to 1:
             the function returns the neighbors of 1 without the node 0

        Parameters:
            - edge: the edge

        Returns:
        A list of neighbors excluded the predecessor node. 
        From the example it would be the node 0.
        """
        G = type(self).G

        prev_node, cur_node = edge
        neighbors = list(G.neighbors(cur_node))
        neighbors.remove(prev_node)

        return neighbors

    def explore(self):
        G = type(self).G

        prev_node, cur_node = self._cur_edge

        neighbors = self._node_neighbors(self._cur_edge)

        # We reached a tip
        if G.nodes[cur_node]['node_type'] == PointType.TIP:
            return False

        res = [(n, G.edges[(cur_node, n)]['walked'], self._compare_neighbor(cur_node, n)) for n in neighbors]

        # There's only one neighbor node: it can only go to that
        if len(neighbors) == 1:
            node, _, _ = res[0]
            self._add_edge((cur_node, node))

            return True

        # There are more neighbors

        # Get the neighbor with minimum angle difference
        min_edge, walked, angle = min(res, key=lambda n: n[2])
        tmp = list(filter(lambda e: e[1] == False, res))  # List of non walked neighbors edges
        if len(tmp) > 0:
            min_edge1, _, angle1 = min(tmp, key=lambda e: e[2])
            if angle1 < 55:
                min_edge = min_edge1

        self._add_edge((cur_node, min_edge))

        return True
