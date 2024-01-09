import glfw
import numpy as np

'''
O------------------------------------------------------------------------------O
| GLFW CLOCK CLASS                                                             |
O------------------------------------------------------------------------------O
'''

class ClockGLFW:

    def __init__(self):
        self._current_time = glfw.get_time()
        self._previous_time = self._current_time
        self._frame_time = 1.0
        self._delta_time = 0.0
        self._update_interval = 0.2
        self._num_frames = 0

    @property
    def frame_time(self):
        return self._frame_time

    @property
    def frame_rate(self):
        return 1.0 / self._frame_time

    @property
    def update_interval(self):
        return self._update_interval

    @update_interval.setter
    def update_interval(self, update_interval: float):
        self._update_interval = float(update_interval)

    def update(self):
        self._num_frames += 1
        self._current_time = glfw.get_time()
        self._delta_time += self._current_time - self._previous_time
        self._previous_time = self._current_time
        if self._delta_time >= self._update_interval:
            self._frame_time = self._delta_time / self._num_frames
            self._delta_time = 0.0
            self._num_frames = 0

    def show_frame_rate(self, glfw_window):
        frame_rate = int(np.round(1.0 / self._frame_time))
        glfw.set_window_title(glfw_window, f'Frame rate : {frame_rate} FPS')

    def show_frame_time(self, glfw_window):
        glfw.set_window_title(glfw_window, f'Frame time : {self._frame_time:.6f} s')
