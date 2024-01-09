import tripy
import numpy as np
from OpenGL.GL import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Import Python modules
from .canvas import Canvas
from .framebuffer import Framebuffer

''' 
O------------------------------------------------------------------------------O
| RENDER CANVAS CLASS FOR ADVANCED WINDOW DRAWING AREA HANDLING                |
O------------------------------------------------------------------------------O
'''

class RenderCanvas(Canvas):

    def __init__(self, pos=(0, 0), size=(400, 300), pix_scale=1.0, range_x=(-1.0, 1.0)):

        # Initialize empty variables
        self._framebuffer = {}
        self._polygon = None
        self._polygon_vao = None
        self._polygon_buffer = None
        self._polygon_buffer_n_indices = None

        # Derived class variables
        self._canvas_pos = np.asarray(pos).astype('int')
        self._canvas_size = np.asarray(size).astype('int')
        self._pix_scale = float(pix_scale)

        # Initialize base class
        render_size = (self._canvas_size / self._pix_scale).astype('int')
        super().__init__(render_size, range_x)

        # Initialize textured polygon points
        self._polygon_points = np.asarray([[0, 0], [0, render_size[1]], render_size, [render_size[0], 0]])


    # O------------------------------------------------------------------------------O
    # | PUBLIC - GETTERS AND SETTERS                                                 |
    # O------------------------------------------------------------------------------O

    @property
    def pos(self):
        return self._canvas_pos

    @pos.setter
    def pos(self, pos):
        self._canvas_pos = np.asarray(pos).astype('int')

    @property
    def size(self):
        return self._canvas_size

    @property
    def pix_scale(self):
        return self._pix_scale

    @property
    def render_size(self):
        return self._size

    @property
    def framebuffer(self):
        return self._framebuffer

    @property
    def mouse_pos(self):
        return self._mouse_pos

    @mouse_pos.setter
    def mouse_pos(self, pos):
        self._mouse_pos = (np.asarray(pos) - self._canvas_pos) / self._pix_scale

    @property
    def is_active(self):
        point = Point(self.s2gl(self._mouse_pos))
        return self._polygon.contains(point)


    # O------------------------------------------------------------------------------O
    # | PUBLIC - FRAMEBUFFER MANIPULATION                                            |
    # O------------------------------------------------------------------------------O

    def init(self):
        if self._polygon_buffer is not None:
            raise ValueError('RenderCanvas is already initialized!')
        # Set polygon
        self._polygon = Polygon(self.s2gl(self._polygon_points.T).T)
        # Create polygon buffer and assign number of indices
        polygon_buffer_array = self._create_polygon_buffer_array()
        self._polygon_buffer_n_indices = polygon_buffer_array.shape[0]
        # Set polygon VBO and VAO
        self._polygon_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._polygon_buffer)
        glBufferData(GL_ARRAY_BUFFER, polygon_buffer_array.nbytes, polygon_buffer_array, GL_DYNAMIC_DRAW)
        self._polygon_vao = glGenVertexArrays(1)
        glBindVertexArray(self._polygon_vao)
        # Enable VAO attributes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

    def set_visible_polygon(self, polygon_points):
        if self._polygon_buffer is None:
            raise ValueError('RenderCanvas is not initialized!')
        self._polygon_points = np.asarray(polygon_points)
        self._polygon = Polygon(self.s2gl(self._polygon_points.T).T)
        # Create polygon buffer and assign number of indices
        polygon_buffer_array = self._create_polygon_buffer_array()
        self._polygon_buffer_n_indices = polygon_buffer_array.shape[0]
        # Update buffer
        glBindBuffer(GL_ARRAY_BUFFER, self._polygon_buffer)
        glBufferData(GL_ARRAY_BUFFER, polygon_buffer_array.nbytes, polygon_buffer_array, GL_DYNAMIC_DRAW)

    def delete(self):
        if self._polygon_buffer is None:
            raise ValueError('RenderCanvas is not initialized!')
        # Delete textured polygon buffers
        glDeleteBuffers(1, [self._polygon_buffer])
        glDeleteVertexArrays(1, [self._polygon_vao])
        # Remove framebuffers
        for fbo in self._framebuffer.values():
            fbo.delete()
        # Reset variables
        self._framebuffer = {}
        self._polygon = None
        self._polygon_vao = None
        self._polygon_buffer = None
        self._polygon_buffer_n_indices = None

    def add_framebuffer(self, fbo_name, gl_internalformat, gl_format, gl_type):
        if fbo_name in self._framebuffer.keys():
            self._framebuffer[fbo_name].delete()
        self._framebuffer[fbo_name] = Framebuffer(self.render_size, gl_internalformat, gl_format, gl_type)
        self._framebuffer[fbo_name].init()

    def delete_framebuffer(self, fbo_name):
        fbo = self._framebuffer.pop(fbo_name)
        fbo.delete()

    def resize(self, size, pix_scale):
        mouse_pos_w_start = self.s2w(self._mouse_pos)
        # Update class variables
        self._canvas_size = np.asarray(size).astype('int')
        self._pix_scale = float(pix_scale)
        # Update base class variables
        render_size = (self._canvas_size / self._pix_scale).astype('int')
        super().resize(render_size)
        self._mouse_pos = self.w2s(mouse_pos_w_start)
        # Update framebuffers
        for fbo in self._framebuffer.values():
            fbo.size = self._size
            fbo.update()


    # O------------------------------------------------------------------------------O
    # | PRIVATE - FRAMEBUFFER MANIPULATION                                           |
    # O------------------------------------------------------------------------------O

    def _create_polygon_buffer_array(self):
        # Polygon triangulation from points in screen coordinates
        triangle_vertices = np.asarray(tripy.earclip(self._polygon_points))
        triangle_vertices = np.squeeze(triangle_vertices.reshape((1, -1, 2)))
        # Create texture coordinates
        triangle_vertices_gl = self.s2gl(triangle_vertices.T).T
        triangle_texture_vertices = triangle_vertices / self._size
        triangle_texture_vertices[:, 1] = 1.0 - triangle_texture_vertices[:, 1]  # Flip texture along y-axis
        return np.hstack((triangle_vertices_gl, triangle_texture_vertices)).astype('float32')
