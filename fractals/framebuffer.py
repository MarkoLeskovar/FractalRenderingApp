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
        self._fbo_id = None
        self._tex_id = None

    @property
    def numpy_type(self):
        return gl_type_to_numpy_type(self.gl_type.name)

    @property
    def num_channels(self):
        return gl_format_to_num_channels(self.gl_format.name)

    @property
    def id(self):
        return self._fbo_id

    @property
    def tex_id(self):
        return self._tex_id

    def init(self):
        if self._fbo_id is not None:
            raise ValueError('Framebuffer is already initialized!')
        # Create a framebuffer object
        self._fbo_id = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_id)
        # Create a texture for a framebuffer
        self._tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, self.gl_internalformat, self.size[0], self.size[1], 0,
                     self.gl_format, self.gl_type, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex_id, 0)

    def update(self):
        if self._fbo_id is None:
            raise ValueError('Framebuffer is not initialized!')
        glBindTexture(GL_TEXTURE_2D, self._tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, self.gl_internalformat, self.size[0], self.size[1], 0,
                     self.gl_format, self.gl_type, None)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo_id)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._tex_id, 0)

    def delete(self):
        if self._fbo_id is None:
            raise ValueError('Framebuffer is not initialized!')
        glDeleteFramebuffers(1, [self._fbo_id])
        glDeleteTextures(1, [self._tex_id])
        self._fbo_id = None
        self._tex_id = None


'''
O------------------------------------------------------------------------------O
| AUXILIARY FUNCTIONS                                                          |
O------------------------------------------------------------------------------O
'''

def gl_format_to_num_channels(gl_format):
    values_dict = {
        'GL_RED':           1,
        'GL_RG':            2,
        'GL_RGB':           3,
        'GL_BGR':           3,
        'GL_RGBA':          4,
        'GL_BGRA':          4,
        'GL_RED_INTEGER':   1,
        'GL_RG_INTEGER':    2,
        'GL_RGB_INTEGER':   3,
        'GL_BGR_INTEGER':   3,
        'GL_RGBA_INTEGER':  4,
        'GL_BGRA_INTEGER':  4
    }
    return values_dict[str(gl_format)]


def gl_type_to_numpy_type(gl_type):
    values_dict = {
        'GL_UNSIGNED_BYTE':  'uint8',
        'GL_BYTE':           'int8',
        'GL_UNSIGNED_SHORT': 'uint16',
        'GL_SHORT':          'int16',
        'GL_UNSIGNED_INT':   'uint32',
        'GL_INT':            'int32',
        'GL_HALF_FLOAT':     'float16',
        'GL_FLOAT':          'float32'
    }
    return values_dict[str(gl_type)]
