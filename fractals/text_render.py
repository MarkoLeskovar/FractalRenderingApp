import os
import numba
import freetype
import numpy as np

# OpenGL modules
from OpenGL.GL import *

# Add python modules
from .shader_utils import create_shader_program, read_shader_source, get_uniform_locations


# TODO : Make the code nicer and move everything to the interactive app !!

PATH_SHADERS = os.path.join(os.path.dirname(__file__), 'shaders')

'''
O------------------------------------------------------------------------------O
| TEXT RENDERING CLASS                                                         |
O------------------------------------------------------------------------------O
'''

class TextRender:

    def __init__(self):

        # Determine maximum number of instances from maximum uniform block size
        self.max_instances = int(glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE) / 64)  # mat4 -> 64 bytes

        # Read shader source code
        vertex_shader_source = read_shader_source(os.path.join(PATH_SHADERS, 'text_render.vert'))
        fragment_shader_source = read_shader_source(os.path.join(PATH_SHADERS, 'text_render.frag'))
        # Dynamically modify the shader source code before compilation
        vertex_shader_source = vertex_shader_source.replace('INSERT_NUM_INSTANCES', str(self.max_instances))
        fragment_shader_source = fragment_shader_source.replace('INSERT_NUM_INSTANCES', str(self.max_instances))
        # Create a shader program
        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        glUseProgram(self.shader_program)

        # Disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # Get uniform locations
        self.uniform_locations = get_uniform_locations(self.shader_program, ['proj_mat', 'color'])

        # Set buffers
        self._set_vertex_buffer()
        self._set_trans_mat_buffer()
        self._set_char_id_buffer()

        # Initialize empty variables
        self.characters = {}
        self.font_texture_array = None



    # O------------------------------------------------------------------------------O
    # | PUBLIC - TEXT MANIPULATION FUNCTIONS                                         |
    # O------------------------------------------------------------------------------O

    def SetWindowSize(self, size):
        self.window_size = np.asarray(size).astype('int')
        self.proj_mat = ortho_transform_mat(0, self.window_size[0], 0, self.window_size[1], -1, 1)


    def SetFont(self, font_type, font_size):
        self.font_size = int(font_size)

        # Make the texture a bit bigger than font size just in case
        self.font_texture_size = int(1.2 * font_size)

        # Number of ASCII characters
        num_ASCII_char = 256

        # Load freetype characters
        face = freetype.Face(font_type)
        face.set_pixel_sizes(self.font_size, self.font_size)

        # Delete previous texture if it exists
        if self.font_texture_array is not None:
            self.characters = {}
            glDeleteTextures(1, [self.font_texture_array])

        # Generate texture array
        self.font_texture_array = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.font_texture_array)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R8, self.font_texture_size, self.font_texture_size, num_ASCII_char, 0, GL_RED, GL_UNSIGNED_BYTE, None)

        # Set texture options
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Load ASCII characters
        for i in range(num_ASCII_char):
            # Load the character glyph
            face.load_char(chr(i), freetype.FT_LOAD_RENDER)
            # Get character size and data
            face_width = face.glyph.bitmap.width
            face_height = face.glyph.bitmap.rows
            face_buffer = face.glyph.bitmap.buffer
            # Update the 3D image
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, face_width, face_height, 1, GL_RED, GL_UNSIGNED_BYTE, face_buffer)
            # Store character for later use
            self.characters[chr(i)] = CharacterSlot(i, face.glyph)


    def DrawText(self, text, x, y, scale, color):
        color = np.asarray(color) / 255.0

        # Activate text rendering
        glUseProgram(self.shader_program)
        glUniform3fv(self.uniform_locations['color'], 1, color.astype('float32'))
        glUniformMatrix4fv(self.uniform_locations['proj_mat'], 1, GL_FALSE, self.proj_mat.astype('float32'))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.font_texture_array)
        glBindVertexArray(self.texture_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)

        # Enable text blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Flip y-axis for top-left origin
        y = self.window_size[1] - y - self.font_size * scale

        # Loop over the text
        x_start = x
        instance_id = 0
        for c in text:

            # Get the current character
            ch = self.characters[c]

            # Check if we have a new line character
            if (c == '\n'):
                x = x_start
                y -= self.font_size * scale

            # Check if we have an empty space or a tab
            elif (c == ' ') or (c == '\t'):
                x += (ch.advance >> 6) * scale

            # Render the character
            else:
                # Get dimensions
                x_pos = x + ch.bearing[0] * scale
                y_pos = y - (self.font_size - ch.bearing[1]) * scale
                size = self.font_texture_size * scale

                # Set up the texture data
                self.trans_mat_array[instance_id, :, :] = char_transform_mat(x_pos, y_pos, size)
                self.char_id_array[instance_id, 0] = ch.ascii_id

                # Advance x-position for next glyph
                x += (ch.advance >> 6) * scale
                # Update the working index
                instance_id += 1

                # Intermediate draw call
                if (instance_id == self.max_instances):
                    self._draw_call(instance_id)
                    instance_id = 0

        # Final draw call
        self._draw_call(instance_id)

        # Disable text blending
        glDisable(GL_BLEND)


    def Delete(self):
        glDeleteBuffers(3, [self.texture_vbo, self.trans_mat_buffer, self.char_id_buffer])
        glDeleteVertexArrays(1, [self.texture_vao])
        glDeleteTextures(1, [self.font_texture_array])
        glDeleteProgram(self.shader_program)


    # O------------------------------------------------------------------------------O
    # | PRIVATE - OPENGL SET FUNCTIONS                                               |
    # O------------------------------------------------------------------------------O

    def _set_vertex_buffer(self):
        # Initialize the data
        vertex_data = np.asarray([0, 1, 0, 0, 1, 1, 1, 0]).astype('float32')
        # Set textured quad VAO and VBO
        self.texture_vao = glGenVertexArrays(1)
        glBindVertexArray(self.texture_vao)
        self.texture_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        # Enable VAO attributes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))


    def _set_trans_mat_buffer(self):
        # Initialize the data
        self.trans_mat_array = np.zeros(shape=(self.max_instances, 4, 4), dtype='float32')
        # Set transformation matrices buffer
        self.trans_mat_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.trans_mat_buffer)
        glBufferData(GL_UNIFORM_BUFFER, self.trans_mat_array.nbytes, None, GL_DYNAMIC_DRAW)
        # Set uniform block binding location
        trans_mat_buffer_block_index = glGetUniformBlockIndex(self.shader_program, 'trans_mat_buffer')
        glUniformBlockBinding(self.shader_program, trans_mat_buffer_block_index, 0)


    def _set_char_id_buffer(self):
        # Initialize the data
        self.char_id_array = np.zeros(shape=(self.max_instances, 4), dtype='int32')
        # Set character ids buffer
        self.char_id_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.char_id_buffer)
        glBufferData(GL_UNIFORM_BUFFER, self.char_id_array.nbytes, None, GL_DYNAMIC_DRAW)
        # Set uniform block binding location
        char_id_buffer_block_index = glGetUniformBlockIndex(self.shader_program, 'char_id_buffer')
        glUniformBlockBinding(self.shader_program, char_id_buffer_block_index, 1)


    # O------------------------------------------------------------------------------O
    # | PRIVATE - OPENGL DRAW CALL                                                   |
    # O------------------------------------------------------------------------------O

    def _draw_call(self, num_instances):
        # Update transformation buffer
        temp_data = self.trans_mat_array[0: num_instances, :, :]
        glBindBuffer(GL_UNIFORM_BUFFER, self.trans_mat_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, temp_data.nbytes, temp_data)
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.trans_mat_buffer)
        # Update character ID buffer
        temp_data = self.char_id_array[0: num_instances, :]
        glBindBuffer(GL_UNIFORM_BUFFER, self.char_id_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, temp_data.nbytes, temp_data)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.char_id_buffer)
        # Draw instanced characters
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, num_instances)


'''
O------------------------------------------------------------------------------O
| CHARACTER SLOT CLASS                                                         |
O------------------------------------------------------------------------------O
'''

class CharacterSlot:

    def __init__(self, ascii_id, glyph):
        self.ascii_id = ascii_id                              # ID of the ASCII character
        self.size = (glyph.bitmap.width, glyph.bitmap.rows)   # Size of glyph
        self.bearing = (glyph.bitmap_left, glyph.bitmap_top)  # Offset from the baseline to left/top of glyph
        self.advance = glyph.advance.x                        # Offset to advance to next glyph


'''
O------------------------------------------------------------------------------O
| AUXILIARY FUNCTIONS                                                          |
O------------------------------------------------------------------------------O
'''

@numba.njit(cache=True)
def char_transform_mat(x_pos, y_pos, size):
    trans_mat = np.zeros(shape=(4, 4), dtype='float')
    trans_mat[0, 0] = size
    trans_mat[1, 1] = size
    trans_mat[3, 0] = x_pos
    trans_mat[3, 1] = y_pos
    trans_mat[3, 3] = 1.0
    return trans_mat

@numba.njit(cache=True)
def ortho_transform_mat(left, right, bottom, top, near, far):
    trans_mat = np.zeros(shape=(4, 4), dtype='float')
    trans_mat[0, 0] = 2.0 / (right - left)
    trans_mat[1, 1] = 2.0 / (top - bottom)
    trans_mat[2, 2] = -2.0 / (far - near)
    trans_mat[3, 0] = -(right + left) / (right - left)
    trans_mat[3, 1] = -(top + bottom) / (top - bottom)
    trans_mat[3, 2] = -(far + near) / (far - near)
    trans_mat[3, 3] = 1.0
    return trans_mat
