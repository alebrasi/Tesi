from .splines import Spline

class Plant():
    def __init__(self, id, label, stem, seed=None, roots = None):
        self.id = id
        self.label = label
        self.stem = stem
        self.seed = seed
        self.roots = roots if roots is not None else []

    @property
    def stem_angle(self):
        return self.stem.angle

    def all_roots(self):
        for r in self.roots:
            # Return current primary
            yield r
            # Return all child roots
            for c in r.roots:
                yield c

    def primary_roots(self):
        for r in self.roots:
            # Return only primary
            yield r

    def lateral_roots(self):
        for r in self.roots:
            # Return only child roots
            for c in r.roots:
                yield c

class Root():
    def __init__(self, points, diameters, roots = None, spline_tension = 0.5, spline_knot_spacing = 50):
        self.roots = roots if roots is not None else []
        self.start = points[0]
        self.end = points[-1]
        self.diameters = diameters
        self.spline = Spline(points, spline_tension, spline_knot_spacing)