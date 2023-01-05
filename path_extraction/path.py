import numpy as np
from path_extraction.vector_utils import Vector

class Path:
    """
    Constructs a Path objects
    Parameters:
        points (list): A list of points in (y,x) format that compose the Path
        m (float): Gradient of the lines
        q (float): The y intercept
        res_err (float): The residual error
    """
    def __init__(self, points, m=0, q=0, res_err=-1):
        self._points = points
        self._m = m
        self._q = q
        self._res_err = res_err

    def __lstsq_from_path(self, path_points):
        """
        Fit a line, y = mx + q through the points in the array
        https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
        """
        arr = np.array(path_points)
        y = arr[:, 0]
        x = arr[:, 1]

        A = np.vstack([x, np.ones(len(x))]).T
        sol, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        m, q = sol

        return m, q, residuals

    @property
    def points(self):
        return self._points

    @property
    def line_coeff(self):
        """
        Gets the coefficients of the path.
        """
        return self._m, self._q 

    @property
    def res_err(self):
        """
        Gets the residual error of the path. -1 If doesn't have one
        """
        return self._res_err

    @property
    def endpoint(self):
        return self._points[-1]

    @property
    def penultimate_point(self):
        return self._points[-2] if len(self._points) > 1 else self.endpoint

    @property
    def lenght(self):
        return len(self._points)

    @property
    def vector(self):
        return Vector(self._points[0][::-1], self._points[-1][::-1])

    def get_lstsq_paths(self, max_res_err, min_points, iterations=-1):
        """
        Returns a list of Paths

        Parameters:
            max_res_err (float): The maximum residual error for slope
            min_points (int): Minimum points required to calculate the slope
            iterations (int): How many lines to calculate. 
                              Default=-1 calculate all the slopes in the path 
        """
        tmp_line_path = []
        line_vals = []
        slope_paths = []

        lines = []
        it = 0
        for i, p in enumerate(self._points, 1):

            if (it != -1) and (it == iterations):
                return lines

            tmp_line_path.append(p)

            if (len(tmp_line_path) > min_points) or (i == self.lenght):
                m, q, res = self.__lstsq_from_path(tmp_line_path)

                """
                If the mean residual value is > max_residual_err, the latest point is removed
                and the current line values and relative set of points are added to the overall list
                """
                if len(res) > 0 and (res/self.lenght) > max_res_err:
                    last = tmp_line_path.pop()
                    m, q, _ = self.__lstsq_from_path(tmp_line_path)
                    lines.append(Path(tmp_line_path, m, q))
                    tmp_line_path = [last]
                    it = it + 1

        # Check on the residual path
        if len(tmp_line_path) < min_points and len(lines) > 0:
            if len(lines) > 0:
                tmp = lines.pop()
                tmp_line_path = tmp.points + tmp_line_path

        m, q, _ = self.__lstsq_from_path(tmp_line_path)
        lines.append(Path(tmp_line_path, m, q))

        return lines

    def angle_between(self, path) -> float:
        """
        Returns the angle between this path and the given path
        """
        # Grab the last least squared path
        m, q = self._m, self._q
        s_path_points = self._points
        n_path_points = path.points

        # For now the vector is created by taking the first and last node of the path
        vs = Vector(s_path_points[0], s_path_points[-1])

        vn = Vector(n_path_points[0], n_path_points[-1])
        angle = abs(Vector.angle_between(vs, vn))

        return angle