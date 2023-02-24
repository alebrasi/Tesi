import copy
import networkx as nx
import math

from graph.graph_creation import PointType

flat_map = lambda f, xs: (y for ys in xs for y in f(ys))

class Root:

    G = None

    @staticmethod
    def attach_graph(G):
        Root.G = G

    def __init__(self, plant, start_edge, ema_alpha=0.20):
        self._edges = set()
        self._cur_edge = None
        self._prev_angle_ema = None
        self._ordered_edges = []
        self._ema_alpha = ema_alpha
        self._start_edge = start_edge
        self._plant = plant
        self._add_edge(start_edge)

        self._split_node = None

    @property
    def angle(self):
        return self._prev_angle_ema

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

#    def remove_edge(self, edge):
        #"""
        #Remove an edge from the root

        #Parameters:
            #- edge (tuple) : the edge to remove
        #"""
        #self._edges.remove(edge)

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

        """
        return [ n for n in self._G.neighbors(node) 
                        if (node, n) not in self._edges and 
                            (n, node) not in self._edges ]
        """

    @property
    def highest_non_walked_node(self):
        G = type(self).G
        """
        non_walked_nodes = map(lambda n: (n, G.nodes[n]['pos']),
                                filter(lambda n: n != self._plant._tip_start_node,
                                    flat_map(lambda n: n, (self.non_walked_neighbors(node) 
                                                        for node in self.nodes)
                                    )
                                )
                            )
        """

        non_walked_nodes = []

        for node in self.nodes:
            neighbors = self.non_walked_neighbors(node)
            for neighbor in neighbors:
                if neighbor == self._plant._stem_start_node:
                    continue
                non_walked_nodes.append((node, neighbor, G.nodes[node]['pos']))

        #non_walked_nodes = list(non_walked_nodes)
        #print(id(self), " ", non_walked_nodes)
        #print(non_walked_nodes)

        return min(non_walked_nodes,
                    key=lambda n: n[2][0],  # key = y coordinate of 'pos'
                    default=None) #[0]                   # TODO: Sistemare.  Grabs only the node


    def copy_until_node(self, node, node_neighbor):
        """
            copied = copy.deepcopy(self)
            tmp_edges = [ True if edge[1] == node else False for edge in copied._ordered_edges ]
            #print(id(self))
            
            #print(node)
            #print(copied._ordered_edges)
            copied._split_node = (node, node_neighbor)

            idx = tmp_edges.index(True)
            copied._cur_edge = copied._ordered_edges[idx]
            copied._edges.difference_update(copied._ordered_edges[idx+1:])
            copied._ordered_edges = copied._ordered_edges[:idx+1]
            #copied._edges.difference_update(tmp_edges[:idx+1])

            # TODO: Scrivere MEGLIO
            # calculate EMA angle
            ema = type(self).G.edges[copied._ordered_edges[0]]['weight']
            for edge in copied._ordered_edges:
                angle = type(self).G.edges[edge]['weight']
                #ema = self._calc_ema(ema, angle, self._ema_alpha)
                copied._update_angle_EMA(edge)

            #copied._prev_angle_ema = ema
            copied._add_edge((node, node_neighbor))

            return copied
        """
        tmp_root = Root(self._plant, self._start_edge)
        tmp_root._split_node = (node, node_neighbor)
        tmp_edges = [ True if edge[1] == node else False for edge in self._ordered_edges ]
        idx = tmp_edges.index(True)
        edges = self._ordered_edges[:idx+1]
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
        #return set(flat_map(lambda e: e[1], self._edges))

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

    def _calc_ema(self, ema, angle, alpha):
        return (alpha * angle) + ((1 - alpha) * ema)

    def _map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _update_angle_EMA(self, edge):
        """
        # TODO Riscrivere meglio la doc
        Calculate the exponential moving average of the angle
        https://www.investopedia.com/terms/e/ema.asp
        https://towardsdatascience.com/moving-averages-in-python-16170e20f6c

        Versione utilizzata:
        https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/
        """
        # TODO: Se funziona, rinominare self._prev_angle_ema in self._angle_ema
        
        G = type(self).G
        #cur_angle = G.edges[self._cur_edge]['weight']
        cur_angle = G.edges[edge]['weight']
        #ema_alpha = G.edges[edge]['length'] / G.graph['max_edge_len']
        
        edge_len = G.edges[edge]['length']
        ema_alpha = 0.8
        if edge_len < 25:
            ema_alpha = self._map_range(edge_len, 1, 25, 0.01, 0.8)

        if self._prev_angle_ema == None:
            self._prev_angle_ema = cur_angle
            return cur_angle

        """
        cur_angle_ema = (self._ema_alpha * cur_angle) +                 \
                        (1-self._ema_alpha) * self._prev_angle_ema
        """
        #cur_angle_ema = self._calc_ema(self._prev_angle_ema, cur_angle, self._ema_alpha)

        cur_angle_ema = self._calc_ema(self._prev_angle_ema, cur_angle, ema_alpha)
        self._prev_angle_ema = cur_angle_ema
        
        #return cur_angle_ema

    def _compare_neighbor(self, node, neighbor):
        G = type(self).G
        # TODO: mettere tutti i seed
        if neighbor == self._plant._stem_start_node:
            return float('inf')

        edge = (node, neighbor)
        return math.fabs(self._prev_angle_ema - G.edges[edge]['weight'])

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
        # Ti prego, scriviti da solo. Non ce la faccio più 😫

        G = type(self).G

        prev_node, cur_node = self._cur_edge

        neighbors = self._node_neighbors(self._cur_edge)

        # We reached a tip
        if G.nodes[cur_node]['node_type'] == PointType.TIP:
        #if len(neighbors) == 0:
            return False

        res = [ (n, G.edges[(cur_node, n)]['walked'], self._compare_neighbor(cur_node, n)) for n in neighbors ]

        # There's only one neighbor node: it can only go to that
        if len(neighbors) == 1:
            node, _, _ = res[0]
            self._add_edge((cur_node, node))

            return True

        # There are more neighbors

        # Get the neighbor with minimum angle difference
        min_edge, walked, angle = min(res, key=lambda n: n[2])
        tmp = list(filter(lambda e: e[1] == False, res))    # List of non walked neighbors edges
        if len(tmp) > 0:
            min_edge1, _, angle1 = min(tmp, key=lambda e: e[2])
            #if math.fabs(angle1 - angle) < 45:
            #val1 = G.edges[(cur_node, min_edge)]['weight']
            #val2 = G.edges[(cur_node, min_edge1)]['weight']

            if angle1 < 55: #and self.is_right_quadrant(val1, val2):
                min_edge = min_edge1
        
        self._add_edge((cur_node, min_edge))

        return True

    def is_right_quadrant(self, from_angle, to_angle):
        q1 = self.get_quadrant(from_angle)
        q2 = self.get_quadrant(to_angle)
        if ((q1 == 1) or (q1 == 2)) and ((q2 == 1) or (q2 == 2)):
            return True
        
        if ((q1 == 3) or (q1 == 4)) and ((q2 == 3) or (q2 == 4)):
            return True

        return False

    def get_quadrant(self, angle):
        """
        Returns the quadrant (1, 2, 3, or 4) of the given angle in degrees.
        """
        if angle < 0 or angle > 360:
            raise ValueError("Angle must be between 0 and 360 degrees.")
        elif angle == 0 or angle == 360:
            return 1
        elif angle > 0 and angle < 90:
            return 1
        elif angle == 90:
            return 2
        elif angle > 90 and angle < 180:
            return 2
        elif angle == 180:
            return 3
        elif angle > 180 and angle < 270:
            return 3
        elif angle == 270:
            return 4
        else:
            return 4
