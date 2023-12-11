import tripy
import numpy as np
from OpenGL.GL import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Import Python modules
from .canvas import Canvas
from .framebuffer import Framebuffer


# TODO : Add RenderCanvasManager class which should act as in interface between window and all canvases. It should also
#      : have a function that return the currently active canvas
# TODO : Add drawing area polygon creation to the class. Input should be in pixels and initialized with full window.
# TODO : Once this is done, merge the changes with the main branch before doing more work.

''' 
O------------------------------------------------------------------------------O
| RENDER CANVAS CLASS FOR ADVANCED WINDOW DRAWING AREA HANDLING                |
O------------------------------------------------------------------------------O
'''

class RenderCanvas(Canvas):

    def __init__(self, pos=(0, 0), size=(400, 300), pix_scale=1.0, range_x=(-1, 1), zoom_min=0.5, zoom_max=1.0e15, zoom_step=0.02):

        # Initialize empty variables
        self._framebuffer = {}
        self._polygon = None
        self._polygon_points = None
        self._polygon_vao = None
        self._polygon_buffer = None
        self._polygon_buffer_n_indices = None

        # Derived class variables
        self._canvas_pos = np.asarray(pos).astype('int')
        self._canvas_size = np.asarray(size).astype('int')
        self._pix_scale = float(pix_scale)

        # Initialize base class
        render_size = (self._canvas_size / self._pix_scale).astype('int')
        super().__init__(render_size, range_x, zoom_min, zoom_max, zoom_step)

        # Initialize a textured polygon
        temp_points = np.asarray([[0, 0], [0, self._size[1]], self._size, [self._size[0], 0]])
        self._set_polygon_buffer(temp_points)


    # O------------------------------------------------------------------------------O
    # | PUBLIC - GETTERS AND SETTERS                                                 |
    # O------------------------------------------------------------------------------O

    @property
    def pos(self):
        return self._canvas_pos

    @property
    def size(self):
        return self._canvas_size

    @property
    def render_size(self):
        return self._size

    @property
    def framebuffer(self):
        return self._framebuffer


    # O------------------------------------------------------------------------------O
    # | PUBLIC - FRAMEBUFFER MANIPULATION                                            |
    # O------------------------------------------------------------------------------O

    def add_framebuffer(self, fbo_name, gl_internalformat, gl_format, gl_type):
        if fbo_name in self._framebuffer.keys():
            self._framebuffer[fbo_name].delete()
        else:
            self._framebuffer[fbo_name] = Framebuffer(self.render_size, gl_internalformat, gl_format, gl_type)


    def delete_framebuffer(self, fbo_name):
        fbo = self._framebuffer.pop(fbo_name)
        fbo.delete()


    def _set_polygon_buffer(self, points_s):
        self._polygon_points = np.asarray(points_s)
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


    def update_polygon_buffer(self, points_s):
        self._polygon_points = np.asarray(points_s)
        self._polygon = Polygon(self.s2gl(self._polygon_points.T).T)
        # Create polygon buffer and assign number of indices
        polygon_buffer_array = self._create_polygon_buffer_array()
        self._polygon_buffer_n_indices = polygon_buffer_array.shape[0]
        # Update buffer
        glBindBuffer(GL_ARRAY_BUFFER, self._polygon_buffer)
        glBufferData(GL_ARRAY_BUFFER, polygon_buffer_array.nbytes, polygon_buffer_array, GL_DYNAMIC_DRAW)


    def _create_polygon_buffer_array(self):
        # Polygon triangulation from points in screen coordinates
        triangle_vertices = np.asarray(tripy.earclip(self._polygon_points))
        triangle_vertices = np.squeeze(triangle_vertices.reshape((1, -1, 2)))
        # Create texture coordinates
        triangle_vertices_gl = self.s2gl(triangle_vertices.T).T
        triangle_texture_vertices = triangle_vertices / self._size
        triangle_texture_vertices[:, 1] = 1.0 - triangle_texture_vertices[:, 1]  # Flip texture along y-axis
        return np.hstack((triangle_vertices_gl, triangle_texture_vertices)).astype('float32')


    def is_active(self):
        point = Point(self.s2gl(self._mouse_pos))
        return self._polygon.contains(point)


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


    def set_mouse_pos(self, pos):
        temp_mp_s = (np.asarray(pos) - self._canvas_pos) / self._pix_scale
        self.mouse_pos = temp_mp_s


    def delete(self):
        glDeleteBuffers(1, [self._polygon_buffer])
        glDeleteVertexArrays(1, [self._polygon_vao])
        # Remove framebuffers
        for fbo in self._framebuffer.values():
            fbo.delete()
