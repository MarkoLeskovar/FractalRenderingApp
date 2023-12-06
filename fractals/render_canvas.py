import numpy as np
from OpenGL.GL import *

# Import Python modules
from .canvas import Canvas


# TODO : Refactor this class at the end to make it more logical

class Framebuffer:

    def __init__(self, size, gl_internalformat, gl_format, gl_type):
        self.size = np.asarray(size).astype('int')
        self.gl_internalformat = gl_internalformat
        self.gl_format = gl_format
        self.gl_type = gl_type
        self.fbo = None
        self.tex = None
        self._create()

    def Update(self):
        self.Delete()
        self._create()

    def Delete(self):
        glDeleteFramebuffers(1, [self.fbo])
        glDeleteTextures(1, [self.tex])
        self.fbo = None
        self.tex = None

    def _create(self):
        # Create a framebuffer object
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        # Create a texture for a framebuffer
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, self.gl_internalformat, self.size[0], self.size[1], 0, self.gl_format, self.gl_type, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex, 0)


''' 
O------------------------------------------------------------------------------O
| RENDER CANVAS CLASS FOR ADVANCED WINDOW DRAWING AREA HANDLING                |
O------------------------------------------------------------------------------O
'''


# TODO : Re-thing how the creation and updating of framebufferes should work!

class RenderCanvas(Canvas):

    def __init__(self, size=(400, 300), range_x=(-1, 1), scale_min=0.5, scale_max=1.0e15, scale_step=0.02):
        super().__init__(size, range_x, scale_min, scale_max, scale_step)
        self.framebuffers = {}

    def AddFramebuffer(self, name, gl_internalformat, gl_format, gl_type):
        if name in self.framebuffers.keys():
            self.framebuffers[name].Delete()
        self.framebuffers[name] = Framebuffer(self.size, gl_internalformat, gl_format, gl_type)

    def RemoveFramebuffer(self, name):
        framebuffer = self.framebuffers.pop(name)
        framebuffer.Delete()

    def Resize(self, size):
        super().Resize(size)
        # Update framebuffers
        for framebuffer in self.framebuffers.values():
            framebuffer.size = self.size
            framebuffer.Update()

    def Delete(self):
        for name in self.framebuffers.keys():
             self.framebuffers[name].Delete()
