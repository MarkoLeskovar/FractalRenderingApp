import numpy as np
from OpenGL.GL import *


''' 
O------------------------------------------------------------------------------O
| FRAMEBUFFER CLASS FOR HANDING FRAMEBUFFERS WITH TEXTURE ATTACHMENTS          |
O------------------------------------------------------------------------------O
'''

class Framebuffer:

    def __init__(self, size, gl_internalformat, gl_format, gl_type):
        self.size = np.asarray(size).astype('int')
        self.gl_internalformat = gl_internalformat
        self.gl_format = gl_format
        self.gl_type = gl_type
        self.fbo = None
        self.tex = None
        self._set_framebuffer()

    def update(self):
        self.delete()
        self._set_framebuffer()

    def delete(self):
        glDeleteFramebuffers(1, [self.fbo])
        glDeleteTextures(1, [self.tex])
        self.fbo = None
        self.tex = None

    def _set_framebuffer(self):
        # Create a framebuffer object
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        # Create a texture for a framebuffer
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, self.gl_internalformat, self.size[0], self.size[1], 0, self.gl_format,
                     self.gl_type, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex, 0)
