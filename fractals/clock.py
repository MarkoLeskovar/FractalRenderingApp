import glfw
import numpy as np

'''
O------------------------------------------------------------------------------O
| GLFW CLOCK CLASS                                                             |
O------------------------------------------------------------------------------O
'''

class ClockGLFW:
    def __init__(self):
        self.current_time = glfw.get_time()
        self.previous_time = self.current_time
        self.frame_time = 1.0
        self.delta_time = 0.0
        self.num_frames = 0

    def Update(self):
        self.num_frames += 1
        self.current_time = glfw.get_time()
        self.delta_time += self.current_time - self.previous_time
        self.previous_time = self.current_time
        if (self.delta_time >= 0.2):
            self.frame_time = self.delta_time / self.num_frames
            self.delta_time = 0.0
            self.num_frames = 0

    def ShowFrameRate(self, window):
        glfw.set_window_title(window, f'Frame rate : {int(np.round(1.0 / self.frame_time))} FPS')

    def ShowFrameTime(self, window):
        glfw.set_window_title(window, f'Frame time : {self.frame_time:.6f} s')
