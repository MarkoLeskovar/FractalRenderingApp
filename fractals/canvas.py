import numpy as np

'''
O------------------------------------------------------------------------------O
| CANVAS CLASS FOR WINDOW DRAWING AREA HANDLING                                |
O------------------------------------------------------------------------------O
'''

class Canvas:

    def __init__(self, size=(400, 300), range_x=(-1.0, 1.0), zoom_min=0.5, zoom_max=1.0e15, zoom_step=0.02):
        self._size = np.asarray(size).astype('int')
        self._range_x_default = np.asarray(range_x).astype('float')
        # Scaling settings
        self._scale_rel_min = float(zoom_min)
        self._scale_rel_max = float(zoom_max)
        self._scale_abs_step = float(zoom_step)
        # Default shift, scale and range variables
        self._shift_default, self._scale_abs_default = self._get_shift_and_scale(self._range_x_default, (0.0, 0.0))
        self._shift = self._shift_default.copy()
        self._scale_abs = self._scale_abs_default
        self._range_x, self._range_y = self._get_range_xy()
        # Initialize mouse position
        self._mouse_pos = np.asarray([0, 0], dtype='float')
        self._mouse_pos_previous = self._mouse_pos.copy()
        # Updating
        self._needs_update = True


    # O------------------------------------------------------------------------------O
    # | GETTERS AND SETTERS                                                          |
    # O------------------------------------------------------------------------------O

    @property
    def size(self) -> np.ndarray:
        return self._size

    @property
    def range_x(self) -> np.ndarray:
        return self._range_x

    @property
    def range_y(self) -> np.ndarray:
        return self._range_y

    @property
    def scale_abs(self) -> float:
        return self._scale_abs

    @property
    def scale_rel(self) -> float:
        return self._scale_abs / self._scale_abs_default

    @property
    def scale_step(self) -> float:
        return self._scale_abs_step

    @property
    def mouse_pos(self) -> np.ndarray:
        return self._mouse_pos

    @mouse_pos.setter
    def mouse_pos(self, value):
        self._mouse_pos = np.asarray(value).astype('float')

    @property
    def needs_update(self) -> bool:
        return self._needs_update

    @needs_update.setter
    def needs_update(self, value):
        self._needs_update = bool(value)

    # O------------------------------------------------------------------------------O
    # | PUBLIC - CANVAS UPDATING                                                     |
    # O------------------------------------------------------------------------------O

    def reset_shift_and_scale(self):
        self._shift = self._shift_default.copy()
        self._scale_abs = self._scale_abs_default
        self._range_x, self._range_y = self._get_range_xy()
        self._needs_update = True

    def resize(self, size):
        range_x_previous, range_y_previous = self._get_range_xy()
        self._size = np.asarray(size).astype('int')
        self._shift_default, self._scale_abs_default = self._get_shift_and_scale(self._range_x_default, (0.0, 0.0))
        self._shift, self._scale_abs = self._get_shift_and_scale(range_x_previous, range_y_previous)
        self._range_x, self._range_y = self._get_range_xy()
        self._needs_update = True

    def update_shift(self):
        delta_shift = self._mouse_pos - self._mouse_pos_previous
        self._shift += delta_shift
        self._range_x, self._range_y = self._get_range_xy()
        self._needs_update = True

    def increase_scale(self, scale_step):
        mouse_pos_w_start = self.s2w(self._mouse_pos)  # Starting position for the mouse
        self._scale_abs *= (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self._scale_abs / self._scale_abs_default) > self._scale_rel_max:
            self._scale_abs = self._scale_rel_max * self._scale_abs_default  # Max zoom
        self._shift += self._mouse_pos - self.w2s(mouse_pos_w_start)  # Correct position by panning
        self._range_x, self._range_y = self._get_range_xy()
        self._needs_update = True

    def decrease_scale(self, scale_step):
        mouse_pos_w_start = self.s2w(self._mouse_pos)  # Starting position for the mouse
        self._scale_abs /= (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self._scale_abs / self._scale_abs_default) < self._scale_rel_min:
            self._scale_abs = self._scale_rel_min * self._scale_abs_default  # Min zoom
        self._shift += self._mouse_pos - self.w2s(mouse_pos_w_start)  # Correct position by panning
        self._range_x, self._range_y = self._get_range_xy()
        self._needs_update = True

    def update_mouse_pos(self):
        self._mouse_pos_previous = self._mouse_pos.copy()


    # O------------------------------------------------------------------------------O
    # | PUBLIC - SCREEN-TO-WORLD & WORLD-TO-SCREEN TRANSFORMATIONS                   |
    # O------------------------------------------------------------------------------O

    def s2w(self, points):
        points = np.asarray(points)
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = (points[0] - self._shift[0]) / self._scale_abs
        output_points[1] = (self._size[1] + self._shift[1] - points[1]) / self._scale_abs
        return output_points

    def w2s(self, points):
        points = np.asarray(points)
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = self._shift[0] + points[0] * self._scale_abs
        output_points[1] = self._size[1] + self._shift[1] - points[1] * self._scale_abs
        return output_points

    def s2gl(self, points):
        output_points = np.asarray(points).astype('float')
        output_points = (2.0 * (output_points.T / self._size) - 1.0).T
        output_points[1] *= -1.0
        return output_points

    def gl2s(self, points):
        output_points = np.asarray(points).astype('float')
        output_points[1] *= -1.0
        output_points = (0.5 * (output_points.T + 1.0) * self._size).T
        return output_points

    def w2gl(self, points):
        return self.s2gl(self.w2s(points))

    def gl2w(self, points):
        return self.s2w(self.gl2s(points))


    # O------------------------------------------------------------------------------O
    # | PRIVATE FUNCTIONS                                                            |
    # O------------------------------------------------------------------------------O

    def _get_shift_and_scale(self, range_x, range_y):
        # Compute the scaling factor
        size_x = range_x[1] - range_x[0]
        size_y = range_y[1] - range_y[0]
        pix_size = size_x / self._size[0]
        scale = 1.0 / pix_size
        # Compute the shift
        temp_shift_x = 0.5 * self._size[0]  # Offset by image center
        temp_shift_x -= (range_x[0] + 0.5 * size_x) * scale  # Offset by x-extent
        temp_shift_y = -0.5 * self._size[1]  # Offset by image center
        temp_shift_y += (range_y[0] + 0.5 * size_y) * scale  # Offset by y-extent
        shift = np.asarray([temp_shift_x, temp_shift_y], dtype='float')
        # Return results
        return shift, scale

    def _get_range_xy(self):
        TL_w = self.s2w(np.asarray([0.0, 0.0]))
        BR_W = self.s2w(np.asarray(self._size))
        range_x = np.asarray([TL_w[0], BR_W[0]])
        range_y = np.asarray([BR_W[1], TL_w[1]])
        return range_x, range_y
