import numpy as np
import math

class Vector:
    def __init__(self, p1, p2):
        """
        Constructs a vector object given two points

        Parameters:
            p1: Source point
            p2: Destination point
        """
        self._p1 = np.array(p1)
        self._p2 = np.array(p2)

        v = self._p2 - self._p1
        self._mag = np.linalg.norm(v)
        self._v = v / self._mag

    @property
    def x(self):
        return self._v[0]

    @property
    def y(self):
        return self._v[1]

    @property
    def magnitude(self):
        return self._mag

    @property
    def angle(self):
        angle = math.atan2(self.y, self.x)
        angle = math.degrees(angle)
        return int(angle + 360 * (1 if angle < 0 else 0))

    @staticmethod
    def angle_between(v1, v2) -> float:
        """
        Returns the angle between two vectors
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

        angle = math.atan2(v2.y, v2.x) - math.atan2(v1.y, v1.x)
        angle = math.degrees(angle)

        return angle

    @staticmethod
    def invert(v):
        """
        Returns the input vector with inverted direction
        """
        return Vector(v._p2, v._p1)

