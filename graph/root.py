import networkx as nx

class Root:

    flat_map = lambda f, xs: (y for ys in xs for y in f(ys))
    G = None

    @staticmethod
    def attach_graph(G):
        Root.G = G

    def __init__(self, start_edge, ema_alpha=0.8):
        self._edges = set()
        self._cur_edge = None
        self._prev_angle_ema = None
        self._ema_alpha = ema_alpha
        self._start_edge = start_edge
        self._add_edge(start_edge)

    def _add_edge(self, edge):
        """
        Add an edge to the root

        Parameters:
            - edge (tuple) : tuple that describes the edge
        """
        self._edges.add(edge)
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

        non_walked_nodes = map(lambda n: (n, G.nodes[n]['pos']), 
                                (self.non_walked_neighbors(node) for node in self.nodes))

        return min(non_walked_nodes, 
                    key=lambda n: n[1][0]  # key = y coordinate of 'pos'
                   )[0]                    # Grabs only the node

    @property
    def nodes(self):
        return set(flat_map(lambda e: e, self._edges))

    @property
    def edges(self):
        """
        Returns all the edges in the root
        """
        return self._edges.copy()

    def points(self):
        # TODO
        pass

    @property
    def cur_edge(self):
        return self._cur_edge

    @property
    def _angle_EMA(self):
        """
        # TODO Riscrivere meglio la doc
        Calculate the exponential moving average of the angle
        https://www.investopedia.com/terms/e/ema.asp
        https://towardsdatascience.com/moving-averages-in-python-16170e20f6c

        Versione utilizzata:
        https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/
        """
        
        G = type(self).G
        cur_angle = G.edges[self._cur_edge]['weight']

        if self._prev_angle_ema == None:
            self._prev_angle_ema = cur_angle
            return cur_angle

        cur_angle_ema = (self._ema_alpha * cur_angle) +                 \
                        (1-self._ema_alpha) * self._prev_angle_ema

        self._prev_angle_ema = cur_angle_ema
        return cur_angle_ema

    def explore(self):

        return True