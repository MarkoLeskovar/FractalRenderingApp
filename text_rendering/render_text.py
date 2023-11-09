import os
import numpy as np
import freetype
import glfw
import glfw.GLFW as GLFW_VAR
from OpenGL.GL import *
import numba

# Add python modules
from fractals.interactive_app import ClockGLFW, read_shader_source, create_shader_program, get_uniform_locations


# TODO : Convert to "instanced" rendering as in (https://www.youtube.com/watch?v=S0PyZKX4lyI)
# TODO : Check font size and how does that fit the DPI of the monitor etc...


@numba.njit(cache=True)
def ortho_mat(left, right, bottom, top, near=-1.0, far=1.0):
    ortho_mat = np.zeros((4, 4), dtype='float')
    ortho_mat[0, 0] = 2.0 / (right - left)
    ortho_mat[1, 1] = 2.0 / (top - bottom)
    ortho_mat[2, 2] = -2.0 / (far - near)
    ortho_mat[0, 3] = -(right + left) / (right - left)
    ortho_mat[1, 3] = -(top + bottom) / (top - bottom)
    ortho_mat[2, 3] = -(far + near) / (far - near)
    ortho_mat[3, 3] = 1.0
    return ortho_mat


class CharacterSlot:

    def __init__(self, texture_id, glyph):
        self.texture_id = texture_id                          # ID handle of the glyph texture
        self.size = (glyph.bitmap.width, glyph.bitmap.rows)   # Size of glyph
        self.bearing = (glyph.bitmap_left, glyph.bitmap_top)  # Offset from the baseline to left/top of glyph
        self.advance = glyph.advance.x                        # Offset to advance to next glyph


# TODO : Add a "terminate" function that deletes all buffers

class TextRenderer:

    def __init__(self):

        # Select the font
        self.font_type = r'C:\Windows\Fonts\arial.ttf'

        # Read shader source code
        shaders_path = os.path.join(os.path.abspath(__file__), os.pardir, 'shaders')
        vertex_shader_source = read_shader_source(os.path.join(shaders_path, 'render_text.vert'))
        fragment_shader_source = read_shader_source(os.path.join(shaders_path, 'render_text.frag'))

        # Create a shader program
        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        glUseProgram(self.shader_program)

        # Get uniform locations
        self.uniform_locations = get_uniform_locations(
            self.shader_program, ['text_color', 'proj_mat'])

        # TODO : Move this outside
        # Define a projection matrix
        proj_mat = ortho_mat(0, 1200, 0, 800).T

        # Send projection matrix to the GPU
        glUniformMatrix4fv(self.uniform_locations['proj_mat'], 1, GL_FALSE, proj_mat.astype('float32'))

        # TODO : Move this to a seperate function

        # Load freetype characters
        face = freetype.Face(self.font_type)
        face.set_pixel_sizes(0, 100)

        # Disable byte-alignment restriction
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

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
            self.characters[chr(i)] = CharacterSlot(texture, face.glyph)


        # Configure VAO/VBO for texture quads
        self.texture_vao = glGenVertexArrays(1)
        glBindVertexArray(self.texture_vao)

        self.texture_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)
        glBufferData(GL_ARRAY_BUFFER, 4 * 4 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))


    def render(self, text, x, y, scale, color):
        text_color = (np.asarray(color) / 255).astype('float32')

        # Change this to vec3 directly
        glUniform3fv(self.uniform_locations['text_color'], 1, text_color)

        # Activate the texture
        glActiveTexture(GL_TEXTURE0)

        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Bind vertex array
        glBindVertexArray(self.texture_vao)

        # Loop over all characters in the text
        for c in text:

            # Get current character
            ch = self.characters[c]

            # Get character dimensions
            x_pos = x + ch.bearing[0] * scale
            y_pos = y - (ch.size[1] - ch.bearing[1]) * scale
            w = ch.size[0] * scale
            h = ch.size[1] * scale

            # Get VBO for each character
            vertices = np.asarray([
                x_pos, y_pos,         0, 1,
                x_pos,     y_pos + h, 0, 0,
                x_pos + w, y_pos,     1, 1,
                x_pos + w, y_pos + h, 1, 0
            ], dtype='float32')

            # Render glyph texture over quad
            glBindTexture(GL_TEXTURE_2D, ch.texture_id)
            # Update content of VBO memory
            glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
            # Render quad
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
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
    text = TextRenderer()

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
        text.render('Some random text', 0, 0, 1.0, (255, 100, 100))
        text.render('Some more random text', 120, 150, 1.0, (100, 255, 100))

        # Swap buffers
        glfw.swap_buffers(window)

    # Terminate the app
    glfw.terminate()


# Run main function
if __name__ == '__main__':
    main()