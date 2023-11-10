import os
import numba
import freetype
import numpy as np
import glm
import glfw
import glfw.GLFW as GLFW_VAR
from OpenGL.GL import *

# Add python modules
from fractals.interactive_app import ClockGLFW, read_shader_source, create_shader_program, get_uniform_locations


# TODO : Convert to "instanced" rendering as in (https://www.youtube.com/watch?v=S0PyZKX4lyI)
# TODO : Check font size and how does that fit the DPI of the monitor etc...


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
            self.shader_program, ['text_color', 'proj_mat', 'trans_mat'])

        # Define a projection matrix
        proj_mat = glm.ortho(0, self.framebuffer_size[0], 0, self.framebuffer_size[1])

        # Send projection matrix to the GPU
        glUniformMatrix4fv(self.uniform_locations['proj_mat'], 1, GL_FALSE, glm.value_ptr(proj_mat))

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



    def terminate(self):
        # Delete OpenGL buffers
        glDeleteBuffers(1, [self.texture_vbo])
        glDeleteVertexArrays(1, [self.texture_vao])
        glDeleteProgram(self.shader_program)


    def set_font(self, font_type, font_height):
        self.font_height = font_height

        # Disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # Load freetype characters
        face = freetype.Face(font_type)
        face.set_pixel_sizes(0, font_height)

        # Load the first 128 characters of the ASCII set
        self.characters = {}
        for i in range(0, 128):
            # Load the character glyph
            face.load_char(chr(i), freetype.FT_LOAD_RENDER)

            # Generate a character texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                face.glyph.bitmap.width,
                face.glyph.bitmap.rows,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                face.glyph.bitmap.buffer)

            # Set texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            # Store character for later use
            self.characters[chr(i)] = Character(texture, face.glyph)


    def add_text(self, text, x, y, scale, color):

        # Rescale the color to [0, 1] range
        text_color = (np.asarray(color) / 255).astype('float32')

        # Activate text rendering
        glUseProgram(self.shader_program)
        glUniform3fv(self.uniform_locations['text_color'], 1, text_color)
        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self.texture_vao)

        # Save original x and y
        x_start = x
        y_start = y

        # Loop over all characters in the text
        for c in text:

            # Get current character
            ch = self.characters[c]

            # Check if we have a new line character
            if (c == '\n'):
                y -= self.font_height * scale
                x = x_start
            # Check if we have an empty space or a tab
            elif (c == ' ') or (c == '\t'):
                x += (ch.advance >> 6) * scale
            # Render the character
            else:
                # Get character dimensions
                x_pos = x + ch.bearing[0] * scale
                y_pos = y - (ch.size[1] - ch.bearing[1]) * scale
                w = ch.size[0] * scale
                h = ch.size[1] * scale

                # Create transformation matrix
                transform = glm.translate(glm.mat4(1.0), glm.vec3(x_pos, y_pos, 0.0))
                transform *= glm.scale(glm.mat4(1.0), glm.vec3(w, h, 0.0))

                # Set transformation matrix to the GPU
                glUniformMatrix4fv(self.uniform_locations['trans_mat'], 1, GL_FALSE, glm.value_ptr(transform))

                # # Get VBO for each character
                # vertices = np.asarray([
                #     x_pos,     y_pos,     0, 1,
                #     x_pos,     y_pos + h, 0, 0,
                #     x_pos + w, y_pos,     1, 1,
                #     x_pos + w, y_pos + h, 1, 0
                # ], dtype='float32')

                # Render glyph texture over quad
                glBindTexture(GL_TEXTURE_2D, ch.texture_id)
                # Update content of VBO memory
                glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)
                # glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
                # Render quad
                glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, 1)
                # Advance cursors for next glyph (note that advance is number of 1/64 pixels)
                x += (ch.advance >> 6) * scale  # Bit-shift by 6 to get value in pixels (2^6 = 64)


# Define main function
def main():

    # Define window size
    window_size = (1200, 800)

    # Create a GLFW window
    glfw.init()
    glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(GLFW_VAR.GLFW_DOUBLEBUFFER, GLFW_VAR.GLFW_TRUE)
    window = glfw.create_window(window_size[0], window_size[1], 'Text rendering', None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    # Initialize a clock
    clock = ClockGLFW()

    # Initialize a text rendered
    text_renderer = TextRenderer(window_size)
    text_renderer.set_font(r'C:\Windows\Fonts\arial.ttf', font_height=50)

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
        text_renderer.add_text('Some random text', 0, 0, 1.0, (255, 100, 100))
        text_renderer.add_text('Some \t    more \nrandom \ttext', 120, 150, 1.0, (100, 255, 100))

        # Swap buffers
        glfw.swap_buffers(window)

    # Terminate the app
    text_renderer.terminate()
    glfw.terminate()


# Run main function
if __name__ == '__main__':
    main()