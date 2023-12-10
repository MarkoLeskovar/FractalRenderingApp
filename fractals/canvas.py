import numpy as np


'''
O------------------------------------------------------------------------------O
| CANVAS CLASS FOR WINDOW DRAWING AREA HANDLING                                |
O------------------------------------------------------------------------------O
'''

class Canvas:

    def __init__(self, size=(400, 300), range_x=(-1, 1), zoom_min=0.5, zoom_max=1.0e15, zoom_step=0.02):
        self.size = np.asarray(size).astype('int')
        self.range_x_default = np.asarray(range_x).astype('float')
        # Scaling settings
        self.scale_rel_min = float(zoom_min)
        self.scale_rel_max = float(zoom_max)
        self.scale_abs_step = float(zoom_step)
        # Default shift, scale and range variables
        self.shift_default, self.scale_abs_default = self.get_shift_and_scale(self.range_x_default, (0.0, 0.0))
        self.shift = self.shift_default.copy()
        self.scale_abs = self.scale_abs_default
        self.range_x, self.range_y = self.get_range_xy()
        # Initialize mouse position
        self.mouse_pos = np.asarray([0, 0], dtype='float')
        self.mouse_pos_previous = self.mouse_pos.copy()

    def get_shift_and_scale(self, range_x, range_y):
        # Compute the scaling factor
        size_x = range_x[1] - range_x[0]
        size_y = range_y[1] - range_y[0]
        pix_size = size_x / self.size[0]
        scale = 1.0 / pix_size
        # Compute the shift
        temp_shift_x = 0.5 * self.size[0]  # Offset by image center
        temp_shift_x -= (range_x[0] + 0.5 * size_x) * scale  # Offset by x-extent
        temp_shift_y = -0.5 * self.size[1]  # Offset by image center
        temp_shift_y += (range_y[0] + 0.5 * size_y) * scale  # Offset by y-extent
        shift = np.asarray([temp_shift_x, temp_shift_y], dtype='float')
        # Return results
        return shift, scale

    def get_range_xy(self):
        TL_w = self.s2w(np.asarray([0.0, 0.0]))
        BR_W = self.s2w(np.asarray(self.size))
        range_x = np.asarray([TL_w[0], BR_W[0]])
        range_y = np.asarray([BR_W[1], TL_w[1]])
        return range_x, range_y

    def reset_shift_and_scale(self):
        self.shift = self.shift_default.copy()
        self.scale_abs = self.scale_abs_default
        self.range_x, self.range_y = self.get_range_xy()

    def resize(self, size):
        range_x_previous, range_y_previous = self.get_range_xy()
        self.size = np.asarray(size).astype('int')
        self.shift_default, self.scale_abs_default = self.get_shift_and_scale(self.range_x_default, (0.0, 0.0))
        self.shift, self.scale_abs = self.get_shift_and_scale(range_x_previous, range_y_previous)
        self.range_x, self.range_y = self.get_range_xy()

    def update_shift(self):
        delta_shift = self.mouse_pos - self.mouse_pos_previous
        self.shift += delta_shift
        self.range_x, self.range_y = self.get_range_xy()

    def increase_scale(self, scale_step):
        temp_MP_w_start = self.s2w(self.mouse_pos)  # Starting position for the mouse
        self.scale_abs *= (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self.scale_abs / self.scale_abs_default) > self.scale_rel_max:
            self.scale_abs = self.scale_rel_max * self.scale_abs_default  # Max zoom
        self.shift += self.mouse_pos - self.w2s(temp_MP_w_start)  # Correct position by panning
        self.range_x, self.range_y = self.get_range_xy()

    def decrease_scale(self, scale_step):
        temp_MP_w_start = self.s2w(self.mouse_pos)  # Starting position for the mouse
        self.scale_abs /= (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self.scale_abs / self.scale_abs_default) < self.scale_rel_min:
            self.scale_abs = self.scale_rel_min * self.scale_abs_default  # Min zoom
        self.shift += self.mouse_pos - self.w2s(temp_MP_w_start)  # Correct position by panning
        self.range_x, self.range_y = self.get_range_xy()

    def set_mouse_pos(self, pos):
        self.mouse_pos = np.asarray(pos).astype('float')

    def update_mouse_pos_previous(self):
        self.mouse_pos_previous = self.mouse_pos.copy()

    # O------------------------------------------------------------------------------O
    # | SCREEN-TO-WORLD & WORLD-TO-SCREEN TRANSFORMATIONS                            |
    # O------------------------------------------------------------------------------O

    def s2w(self, points):
        points = np.asarray(points)
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = (points[0] - self.shift[0]) / self.scale_abs
        output_points[1] = (self.size[1] + self.shift[1] - points[1]) / self.scale_abs
        return output_points

    def w2s(self, points):
        points = np.asarray(points)
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = self.shift[0] + points[0] * self.scale_abs
        output_points[1] = self.size[1] + self.shift[1] - points[1] * self.scale_abs
        return output_points

    def s2gl(self, points):
        output_points = np.asarray(points).astype('float')
        output_points = (2.0 * (output_points.T / self.size) - 1.0).T
        output_points[1] *= -1.0
        return output_points

    def gl2s(self, points):
        output_points = np.asarray(points).astype('float')
        output_points[1] *= -1.0
        output_points = (0.5 * (output_points.T + 1.0) * self.size).T
        return output_points

    def w2gl(self, points):
        return self.s2gl(self.w2s(points))

    def gl2w(self, points):
        return self.s2w(self.gl2s(points))
