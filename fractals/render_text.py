import os
import freetype
import numpy as np

# OpenGL modules
from OpenGL.GL import *

# Add python modules
from .shader_utils import  texture_transform_mat, ortho_transform_mat
from .shader_utils import create_shader_program, read_shader_source, get_uniform_locations


'''
O------------------------------------------------------------------------------O
| TEXT RENDERING CLASS                                                         |
O------------------------------------------------------------------------------O
'''

class RenderText:

    # "Static" variable
    path_to_shaders = os.path.join(os.path.dirname(__file__), 'shaders')

    def __init__(self):
        # Initialize empty variables
        self._window_size = None
        self._proj_mat = None
        self._font_size = None
        self._font_texture_size = None
        self._font_texture_array = None
        self._characters = {}
        # Determine maximum number of instances from uniform block size
        self._max_instances = int(glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE) / 64)  # mat4 -> 64 bytes
        # Create shader program
        self._set_shader_program()
        self._set_uniform_locations()
        # Set buffers
        self._set_vertex_buffer()
        self._set_trans_mat_buffer()
        self._set_char_id_buffer()


    # O------------------------------------------------------------------------------O
    # | PUBLIC - GETTERS AND SETTERS                                                 |
    # O------------------------------------------------------------------------------O

    @property
    def window_size(self):
        return self._window_size

    @property
    def font_size(self):
        return self._font_size


    # O------------------------------------------------------------------------------O
    # | PUBLIC - TEXT MANIPULATION FUNCTIONS                                         |
    # O------------------------------------------------------------------------------O

    def set_window_size(self, size):
        self._window_size = np.asarray(size).astype('int')
        self._proj_mat = ortho_transform_mat(0, self._window_size[0], 0, self._window_size[1], -1, 1)


    def set_font(self, font_type, font_size):
        num_ASCII_char = 256
        self._font_size = int(font_size)

        # Disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # Load freetype characters
        face = freetype.Face(font_type)
        face.set_pixel_sizes(self._font_size, self._font_size)

        # Delete previous texture if it exists
        if self._font_texture_array is not None:
            self._characters = {}
            glDeleteTextures(1, [self._font_texture_array])

        # Generate texture array
        self._font_texture_size = int(1.2 * font_size)  # A bit bigger just in case
        self._font_texture_array = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._font_texture_array)
        zero_array = np.zeros(shape=(self._font_texture_size * self._font_texture_size * num_ASCII_char), dtype='uint8')
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R8, self._font_texture_size, self._font_texture_size, num_ASCII_char, 0, GL_RED, GL_UNSIGNED_BYTE, zero_array)

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
            self._characters[chr(i)] = CharacterSlot(i, face.glyph)


    def draw_text(self, text, pos, scale, color):
        pos = np.asarray(pos)
        color = np.asarray(color) / 255.0

        # Activate text rendering
        glViewport(0, 0, self._window_size[0], self._window_size[1])
        glUseProgram(self._shader_program)
        glUniform3fv(self._uniform_locations['color'], 1, color.astype('float32'))
        glUniformMatrix4fv(self._uniform_locations['proj_mat'], 1, GL_FALSE, self._proj_mat.astype('float32'))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._font_texture_array)
        glBindVertexArray(self._texture_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._texture_vbo)

        # Enable text blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Flip y-axis for top-left origin
        pos[1] = self._window_size[1] - pos[1] - self._font_size * scale

        # Loop over the text
        x_start = pos[0]
        instance_id = 0
        for c in text:

            # Get the current character
            ch = self._characters[c]

            # Check if we have a new line character
            if c == '\n':
                pos[0] = x_start
                pos[1] -= self._font_size * scale

            # Check if we have an empty space or a tab
            elif (c == ' ') or (c == '\t'):
                pos[0] += (ch.advance >> 6) * scale

            # Render the character
            else:
                # Get dimensions
                pos_x = pos[0] + ch.bearing[0] * scale
                pos_y = pos[1] - (self._font_size - ch.bearing[1]) * scale
                size = self._font_texture_size * scale

                # Set up the texture data
                self._trans_mat_array[instance_id, :, :] = texture_transform_mat(pos_x, pos_y, size, size)
                self._char_id_array[instance_id, 0] = ch.ascii_id

                # Advance x-position for next glyph
                pos[0] += (ch.advance >> 6) * scale
                # Update the working index
                instance_id += 1

                # Intermediate draw call
                if instance_id == self._max_instances:
                    self._render_call(instance_id)
                    instance_id = 0

        # Final draw call
        self._render_call(instance_id)

        # Disable text blending
        glDisable(GL_BLEND)


    def delete(self):
        glDeleteBuffers(3, [self._texture_vbo, self._trans_mat_buffer, self._char_id_buffer])
        glDeleteVertexArrays(1, [self._texture_vao])
        if self._font_texture_array is not None:
            glDeleteTextures(1, [self._font_texture_array])
        glDeleteProgram(self._shader_program)


    # O------------------------------------------------------------------------------O
    # | PRIVATE - OPENGL SET FUNCTIONS                                               |
    # O------------------------------------------------------------------------------O

    def _set_shader_program(self):
        # Read shader source code
        vertex_shader_source = read_shader_source(os.path.join(self.path_to_shaders, 'text_render.vert'))
        fragment_shader_source = read_shader_source(os.path.join(self.path_to_shaders, 'text_render.frag'))
        # Dynamically modify the shader source code before compilation
        vertex_shader_source = vertex_shader_source.replace('INSERT_NUM_INSTANCES', str(self._max_instances))
        fragment_shader_source = fragment_shader_source.replace('INSERT_NUM_INSTANCES', str(self._max_instances))
        # Create a shader program
        self._shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        glUseProgram(self._shader_program)

    def _set_uniform_locations(self):
        self._uniform_locations = get_uniform_locations(self._shader_program, ['proj_mat', 'color'])

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

    def _set_trans_mat_buffer(self):
        # Initialize the data
        self._trans_mat_array = np.zeros(shape=(self._max_instances, 4, 4), dtype='float32')
        # Set transformation matrices buffer
        self._trans_mat_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self._trans_mat_buffer)
        glBufferData(GL_UNIFORM_BUFFER, self._trans_mat_array.nbytes, None, GL_DYNAMIC_DRAW)
        # Set uniform block binding location
        trans_mat_buffer_block_index = glGetUniformBlockIndex(self._shader_program, 'trans_mat_buffer')
        glUniformBlockBinding(self._shader_program, trans_mat_buffer_block_index, 0)

    def _set_char_id_buffer(self):
        # Initialize the data
        self._char_id_array = np.zeros(shape=(self._max_instances, 4), dtype='int32')
        # Set character ids buffer
        self._char_id_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self._char_id_buffer)
        glBufferData(GL_UNIFORM_BUFFER, self._char_id_array.nbytes, None, GL_DYNAMIC_DRAW)
        # Set uniform block binding location
        char_id_buffer_block_index = glGetUniformBlockIndex(self._shader_program, 'char_id_buffer')
        glUniformBlockBinding(self._shader_program, char_id_buffer_block_index, 1)


    # O------------------------------------------------------------------------------O
    # | PRIVATE - OPENGL DRAW CALL                                                   |
    # O------------------------------------------------------------------------------O

    def _render_call(self, num_instances):
        # Update transformation buffer
        temp_data = self._trans_mat_array[0: num_instances, :, :]
        glBindBuffer(GL_UNIFORM_BUFFER, self._trans_mat_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, temp_data.nbytes, temp_data)
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self._trans_mat_buffer)
        # Update character ID buffer
        temp_data = self._char_id_array[0: num_instances, :]
        glBindBuffer(GL_UNIFORM_BUFFER, self._char_id_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, temp_data.nbytes, temp_data)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self._char_id_buffer)
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
