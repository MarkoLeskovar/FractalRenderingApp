import os
import numba
import numpy as np

# OpenGL modules
from OpenGL.GL import *

# Add python modules
from .shader_utils import texture_transform_mat, ortho_transform_mat
from .shader_utils import create_shader_program, read_shader_source, get_uniform_locations


'''
O------------------------------------------------------------------------------O
| CLASS TO BLIT TEXTURE TO SCREEN                                              |
O------------------------------------------------------------------------------O
'''

class RenderTexture:

    # "Static" variable
    _path_to_shaders = os.path.join(os.path.dirname(__file__), 'shaders')

    def __init__(self):
        pass


    def init(self):
        self._set_shader_program()
        self._set_uniform_locations()
        self._set_vertex_buffer()


    def __call__(self, win_size, tex_pos, tex_size, tex_id):
        win_size = np.asarray(win_size).astype('int')
        tex_pos = np.asarray(tex_pos).astype('int')
        tex_size = np.asarray(tex_size).astype('int')

        # Set texture transformation
        pos_x = tex_pos[0]
        pos_y = win_size[1] - tex_pos[1] - tex_size[1]
        trans_mat = texture_transform_mat(pos_x, pos_y, tex_size[0], tex_size[1])
        proj_mat = ortho_transform_mat(0, win_size[0], 0, win_size[1], -1, 1)

        # Set blending equations
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Set viewport and framebuffer
        glViewport(0, 0, win_size[0], win_size[1])
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glUseProgram(self._shader_program)
        # Send uniforms to the GPU
        glUniformMatrix4fv(self._uniform_locations['proj_mat'], 1, GL_FALSE, proj_mat.astype('float32'))
        glUniformMatrix4fv(self._uniform_locations['trans_mat'], 1, GL_FALSE, trans_mat.astype('float32'))
        # Activate texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        # Draw texture
        glBindVertexArray(self._texture_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._texture_vbo)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        # Disable blending
        glDisable(GL_BLEND)


    def delete(self):
        glDeleteBuffers(1, [self._texture_vbo])
        glDeleteVertexArrays(1, [self._texture_vao])
        glDeleteProgram(self._shader_program)


    # O------------------------------------------------------------------------------O
    # | PRIVATE - OPENGL SET FUNCTIONS                                               |
    # O------------------------------------------------------------------------------O

    def _set_shader_program(self):
        # Read shader source code
        vertex_shader_source = read_shader_source(os.path.join(self._path_to_shaders, 'texture_render.vert'))
        fragment_shader_source = read_shader_source(os.path.join(self._path_to_shaders, 'texture_render.frag'))
        # Create a shader program
        self._shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        glUseProgram(self._shader_program)

    def _set_uniform_locations(self):
        self._uniform_locations = get_uniform_locations(self._shader_program, ['proj_mat', 'trans_mat'])

    def _set_vertex_buffer(self):
        # Initialize the data
        vertex_data = np.asarray([0, 1, 0, 0, 1, 1, 1, 0]).astype('float32')
        # Set textured quad VAO and VBO
        self._texture_vao = glGenVertexArrays(1)
        glBindVertexArray(self._texture_vao)
        self._texture_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._texture_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        # Enable VAO attributes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
