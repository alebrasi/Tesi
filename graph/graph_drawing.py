import math
import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G, 
                node_attribute='pos', 
                edge_attribute='weight', 
                node_color='blue', 
                with_labels=True,
                invert_xaxis=True, 
                node_size=100, rad=0.1):

    reverse_coords = lambda t: t[::-1]

    # Reverses the coordinates of the point (x, y) -> (y, x)
    pos = dict(map(lambda n: (n[0], reverse_coords(n[1])), 
                    nx.get_node_attributes(G, node_attribute).items()))

    plt.rc('font', size=8)          # controls default text sizes
    # Write text on top of the corresponding edge
    for edge in G.edges:
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]

        # Angle between the two nodes
        d = math.degrees(math.atan2(y2-y1, x2-x1))
        """
        print(d)
        tmp = (180+d)+180 if d < 0 else d
        sign = -1 if tmp > 90 and tmp < 270 else 1
        d = tmp + sign * 90
        """

        edge_val = G.edges[edge][edge_attribute]

        x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2.
        dx, dy = x2 - x1, y2 - y1

        f = rad

        # Corresponds to the center between edge 1 and edge 2
        t = 0.5

        # Control point of the bezier curve
        cx, cy = x12 + f * dy, y12 - f * dx

        x, y = quadratic_bezier((x1,y1), (cx, cy), (x2,y2), t)

        t = plt.text(x, y, edge_val, rotation=d, backgroundcolor='white')
        t.set_bbox(dict(facecolor='white', alpha=0.0))

    nx.draw(G, 
            pos, 
            with_labels=with_labels, 
            node_color=node_color, 
            connectionstyle=f'arc3, rad = {rad}', 
            node_size=node_size, 
            font_size=6)

    plt.gca().invert_yaxis()
    if invert_xaxis:
        plt.gca().invert_xaxis()
    plt.show()

def quadratic_bezier(p0, p1, p2, t):
    """
    Computes a point on a quadratic Bezier curve.
    https://it.wikipedia.org/wiki/Curva_di_B%C3%A9zier

    Parameters:
    p0 (tuple): Starting point.
    p1 (tuple): Control point.
    p2 (tuple): End point.
    t (float): Parameter value in the range [0, 1].
    
    Returns:
    tuple: Coordinates of the point on the curve.
    """
    x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return (x, y)