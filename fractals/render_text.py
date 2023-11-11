import os
import numba
import freetype
import numpy as np
import glm
import glfw
import glfw.GLFW as GLFW_VAR
from OpenGL.GL import *

import matplotlib.pyplot as plt

# Add python modules
from fractals.interactive_app import ClockGLFW, read_shader_source, create_shader_program, get_uniform_locations


# TODO : Convert to "instanced" rendering as in (https://www.youtube.com/watch?v=S0PyZKX4lyI)
# TODO : Check font size and how does that fit the DPI of the monitor etc...

ARRAY_LIMIT = 128

class Character:

    def __init__(self, texture_id, glyph):
        self.texture_id = texture_id                          # ID handle of the glyph texture
        self.size = (glyph.bitmap.width, glyph.bitmap.rows)   # Size of glyph
        self.bearing = (glyph.bitmap_left, glyph.bitmap_top)  # Offset from the baseline to left/top of glyph
        self.advance = glyph.advance.x                        # Offset to advance to next glyph


class TextRenderer:

    def __init__(self, framebuffer_size):
        self.framebuffer_size = np.asarray(framebuffer_size).astype('int')

        # Read shader source code
        shaders_path = os.path.join(os.path.abspath(__file__), os.pardir, 'shaders')
        vertex_shader_source = read_shader_source(os.path.join(shaders_path, 'text_render.vert'))
        fragment_shader_source = read_shader_source(os.path.join(shaders_path, 'text_render.frag'))

        # Create a shader program
        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        glUseProgram(self.shader_program)

        # Get uniform locations
        self.uniform_locations = get_uniform_locations(
            self.shader_program, ['text_color', 'proj_mat', 'trans_mat', 'character_ids'])

        # Define a projection matrix
        proj_mat = np.array(glm.ortho(0, self.framebuffer_size[0], 0, self.framebuffer_size[1])).T

        # Send projection matrix to the GPU
        glUniformMatrix4fv(self.uniform_locations['proj_mat'], 1, GL_FALSE, proj_mat)

        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Setup a generic rectangle
        vertex_data = np.asarray([
            0, 1,
            0, 0,
            1, 1,
            1, 0
        ]).astype('float32')


        # Configure VAO/VBO for texture quads
        self.texture_vao = glGenVertexArrays(1)
        glBindVertexArray(self.texture_vao)

        self.texture_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))

        # Setup texture array
        self.trans_mat = np.zeros(shape=(ARRAY_LIMIT, 4, 4), dtype='float32')
        self.character_index = np.zeros(shape=ARRAY_LIMIT, dtype='int32')


    def terminate(self):
        # Delete OpenGL buffers
        glDeleteBuffers(1, [self.texture_vbo])
        glDeleteVertexArrays(1, [self.texture_vao])
        glDeleteTextures(1, [self.texture_array])
        glDeleteProgram(self.shader_program)


    def set_font(self, type, size):
        self.font_size = int(size)

        # Number of ASCII characters
        num_ASCII_char = 256

        # Load freetype characters
        face = freetype.Face(type)
        face.set_pixel_sizes(self.font_size, self.font_size)

        # Disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # Generate texture array
        self.texture_array = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_array)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R8, self.font_size, self.font_size, num_ASCII_char, 0, GL_RED, GL_UNSIGNED_BYTE, None)
        # Set texture options
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Load ASCII characters
        self.characters = {}
        for i in range(0, num_ASCII_char):
            # Load the character glyph
            face.load_char(chr(i), freetype.FT_LOAD_RENDER)
            # Get face data
            face_width = face.glyph.bitmap.width
            face_height = face.glyph.bitmap.rows
            face_buffer = face.glyph.bitmap.buffer
            # Update 3D image
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, face_width, face_height, 1, GL_RED, GL_UNSIGNED_BYTE, face_buffer)
            # Store character for later use
            self.characters[chr(i)] = Character(i, face.glyph)


    def add_text(self, text, x, y, scale, color):

        # Rescale the color to [0, 1] range
        text_color = (np.asarray(color) / 255.0).astype('float32')

        # Activate text rendering
        glUseProgram(self.shader_program)
        glUniform3fv(self.uniform_locations['text_color'], 1, text_color)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_array)
        glBindVertexArray(self.texture_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)

        # Save original x and y
        x_start = x

        # Loop over all characters in the text
        working_index = 0
        for c in text:

            # Get current character
            ch = self.characters[c]

            # Check if we have a new line character
            if (c == '\n'):
                y -= self.font_size * scale
                x = x_start
            # Check if we have an empty space or a tab
            elif (c == ' ') or (c == '\t'):
                x += (ch.advance >> 6) * scale
            # Render the character
            else:
                # Get character dimensions
                x_pos = x + ch.bearing[0] * scale
                y_pos = y - (self.font_size - ch.bearing[1]) * scale

                # Create transformation matrix
                temp_trans_mat = (glm.translate(glm.mat4(1.0), glm.vec3(x_pos, y_pos, 0.0)) *
                                  glm.scale(glm.mat4(1.0), glm.vec3(self.font_size * scale, self.font_size * scale, 0.0)))

                # Setup a texture index
                self.trans_mat[working_index] = np.array(temp_trans_mat).T
                self.character_index[working_index] = ch.texture_id

                # Advance cursors for next glyph
                x += (ch.advance >> 6) * scale

                # Update the working index
                working_index += 1

                # Draw call
                if (working_index == ARRAY_LIMIT):
                    self.draw_arrays(working_index)
                    working_index = 0

        # Final draw call
        self.draw_arrays(working_index)


    def draw_arrays(self, num_instances):
        glUniformMatrix4fv(self.uniform_locations['trans_mat'], num_instances, GL_FALSE, self.trans_mat)
        glUniform1iv(self.uniform_locations['character_ids'], num_instances, self.character_index)
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, num_instances)


# Define main function
def main():

    # Define window size
    window_size = (1600, 1000)

    # Create a GLFW window
    glfw.init()
    glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(GLFW_VAR.GLFW_DOUBLEBUFFER, GLFW_VAR.GLFW_TRUE)
    window = glfw.create_window(window_size[0], window_size[1], 'Text rendering', None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Initialize a clock
    clock = ClockGLFW()

    # Initialize a text rendered
    text_renderer = TextRenderer(window_size)
    text_renderer.set_font(r'C:\Windows\Fonts\arial.ttf', size=50)

    # Main render loop
    while not glfw.window_should_close(window):

        # Pool events
        glfw.poll_events()
        clock.Update()
        clock.ShowFrameRate(window)

        # Clear the current framebuffer
        glClearColor(0.1, 0.1, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Render text
        text_renderer.add_text('Lorem ipsum dolor sit amet, consectetur adipiscing elit,\n'
                               'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n'
                               'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris\n'
                               'nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in\n'
                               'reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla\n'
                               'pariatur. Excepteur sint occaecat cupidatat non proident, sunt in\n'
                               'culpa qui officia deserunt mollit anim id est laborum.',
                               10, 950, 1.0, (255, 0, 0))

        # Swap buffers
        glfw.swap_buffers(window)

    # Terminate the app
    text_renderer.terminate()
    glfw.terminate()


# Run main function
if __name__ == '__main__':
    main()