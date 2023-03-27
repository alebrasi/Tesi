import networkx as nx
import copy
from graph.root import Root


class Plant:
    G = None

    @staticmethod
    def attach_graph(G):
        Plant.G = G

    def __init__(self, seed, stem_start_node):
        self._seed = seed
        self._roots = []
        self._current_root = None
        self._stem = None
        self._stem_start_node = stem_start_node
        self._seed_node_neighbors = list(type(self).G.neighbors(seed))
        self._seed_node_neighbors.remove(stem_start_node)
        type(self).G.edges[(seed, stem_start_node)]['walked'] = True

        self._is_finished = False

        self._find_stem()
        for _ in range(len(self._seed_node_neighbors)):
            new_root = self._init_new_root_starting_from_seed()
            while new_root.explore():
                pass
            self._roots.append(new_root)
        self._seed_node_neighbors = []

    @property
    def has_finished(self):
        return self._is_finished

    @property
    def stem(self):
        return self._stem

    @property
    def seed_coords(self):
        return type(self).G.nodes[self._seed]['pos']

    def _find_stem(self):
        self._stem = Root(self, (self._seed, self._stem_start_node))
        while self._stem.explore():
            continue

    def _init_new_root_starting_from_seed(self):
        n = self._seed_node_neighbors.pop()
        return Root(self, (self._seed, n))

    def _init_new_root(self):
        new_root = None
        if len(self._seed_node_neighbors) == 0:
            # Finds the root that has the highest non walked neighbor
            # TODO: Provare a fare un sort e guardare quello con angolo di inserimento minore
            root = min(
                filter(lambda r: r[1] is not None,
                       map(lambda r: (r, r.highest_non_walked_node),
                           self._roots)
                       ),
                key=lambda k: k[1][2][0],  # Order by y coordinate of highest non walked node of the root
                default=None)

            if root is None:
                return False

            root, highest_node = root
            node, neighbor, pos = highest_node

            print('plant: ', node)

            new_root = root.copy_until_node(node, neighbor)

        self._roots.append(new_root)
        self._current_root = new_root

        return True

    def compute(self):
        if self._is_finished:
            return False

        if not self._init_new_root():
            self._is_finished = True
            return False

        while self._current_root.explore():
            pass

        return True

    @property
    def roots(self):
        return copy.deepcopy(self._roots)
