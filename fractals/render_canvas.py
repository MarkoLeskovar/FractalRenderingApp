import numpy as np
from OpenGL.GL import *

# Import Python modules
from .canvas import Canvas
from .framebuffer import Framebuffer


''' 
O------------------------------------------------------------------------------O
| RENDER CANVAS CLASS FOR ADVANCED WINDOW DRAWING AREA HANDLING                |
O------------------------------------------------------------------------------O
'''

class RenderCanvas(Canvas):

    def __init__(self, size=(400, 300), range_x=(-1, 1), scale_min=0.5, scale_max=1.0e15, scale_step=0.02):
        super().__init__(size, range_x, scale_min, scale_max, scale_step)
        # Initialize empty variables
        self.framebuffers = {}

    def add_framebuffer(self, fbo_name, gl_internalformat, gl_format, gl_type):
        if fbo_name in self.framebuffers.keys():
            self.framebuffers[fbo_name].delete()
        self.framebuffers[fbo_name] = Framebuffer(self.size, gl_internalformat, gl_format, gl_type)

    def remove_framebuffer(self, fbo_name):
        fbo = self.framebuffers.pop(fbo_name)
        fbo.delete()

    def resize(self, size):
        super().resize(size)
        # Update framebuffers
        for fbo in self.framebuffers.values():
            fbo.size = self.size
            fbo.update()

    def delete(self):
        # Remove framebuffers
        for fbo in self.framebuffers.values():
            fbo.delete()
