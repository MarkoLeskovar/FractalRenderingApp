import os
import numpy as np
import freetype
import glfw
import glfw.GLFW as GLFW_VAR
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader
import numba

# Add python modules
from fractals.interactive_app import ClockGLFW


# TODO : Reformat this file as in LearnOpenGL tutorial (https://learnopengl.com/In-Practice/Text-Rendering)
# TODO : Add FPS counter from the interactive_app.py
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



FONT_TYPE = r'C:\Windows\Fonts\arial.ttf'




shader_program = None
Characters = {}
VBO = None
VAO = None


class Character:
    def __init__(self, texture_id, glyph):
        self.texture_id = texture_id                          # ID handle of the glyph texture
        self.size = (glyph.bitmap.width, glyph.bitmap.rows)   # Size of glyph
        self.bearing = (glyph.bitmap_left, glyph.bitmap_top)  # Offset from the baseline to left/top of glyph
        self.advance = glyph.advance.x                        # Offset to advance to next glyph


def read_shader_source(path_to_shader):
    with open(path_to_shader, 'r') as f:
        shader_source = f.read()
    return shader_source


def create_shader_program(vertex_src, fragment_src):
    # Compile the shaders
    vertex_shader = compileShader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_src, GL_FRAGMENT_SHADER)
    # Create a program and link the shaders
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    # Delete the shaders and return the program
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program


shaders_path = os.path.join(os.path.abspath(__file__), os.pardir, 'shaders')
VERTEX_SHADER_SOURCE = read_shader_source(os.path.join(shaders_path, 'render_text.vert'))
FRAGMENT_SHADER_SOURCE = read_shader_source(os.path.join(shaders_path, 'render_text.frag'))



def render_text(text, x, y, scale, color):
    global shader_program
    global Characters
    global VBO
    global VAO

    face = freetype.Face(FONT_TYPE)
    face.set_char_size(48 * 64)

    # Set color
    color_loc = glGetUniformLocation(shader_program, "textColor")
    glUniform3f(color_loc, color[0] / 255, color[1] / 255, color[2] / 255)

    glActiveTexture(GL_TEXTURE0)

    # Enable blending
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glBindVertexArray(VAO)
    for c in text:

        # Get current character
        ch = Characters[c]

        # Get character dimensions
        x_pos = x + ch.bearing[0] * scale
        y_pos = y - (ch.size[1] - ch.bearing[1]) * scale
        w = ch.size[0] * scale
        h = ch.size[1] * scale

        # Get VBO for each character
        vertices = np.asarray([
            x_pos,     y_pos + h, 0, 0,
            x_pos,     y_pos,     0, 1,
            x_pos + w, y_pos,     1, 1,
            x_pos,     y_pos + h, 0, 0,
            x_pos + w, y_pos,     1, 1,
            x_pos + w, y_pos + h, 1, 0], dtype='float32')

        # Render glyph texture over quad
        glBindTexture(GL_TEXTURE_2D, ch.texture_id)
        # Update content of VBO memory
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # Render quad
        glDrawArrays(GL_TRIANGLES, 0, 6)
        # Advance cursors for next glyph (note that advance is number of 1/64 pixels)
        x += (ch.advance >> 6) * scale  # Bitshift by 6 to get value in pixels (2^6 = 64)

    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, 0)



def main():
    global VERTEXT_SHADER
    global FRAGMENT_SHADER_SOURCE
    global shader_program
    global Characters
    global VBO
    global VAO

    # O------------------------------------------------------------------------------O
    # | CREATE GLFW WINDOW                                                           |
    # O------------------------------------------------------------------------------O

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

    # O------------------------------------------------------------------------------O
    # | INITIALIZE TEXT RENDERER                                                     |
    # O------------------------------------------------------------------------------O

    # Create a shader program
    shader_program = create_shader_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
    glUseProgram(shader_program)

    # Define a projection matrix
    proj_mat_location = glGetUniformLocation(shader_program, "proj_mat")
    proj_mat = ortho_mat(0, window_size[0], 0, window_size[1]).T

    # Send projection matrix to the GPU
    glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, proj_mat.astype('float32'))

    # Disable byte-alignment restriction
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    # Load freetype characters
    face = freetype.Face(FONT_TYPE)
    face.set_pixel_sizes(0, 50)

    # Load the first 128 characters of the ASCII set
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
        Characters[chr(i)] = Character(texture, face.glyph)

    glBindTexture(GL_TEXTURE_2D, 0)


    # Configure VAO/VBO for texture quads
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, None)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Initialize a clock
    clock = ClockGLFW()

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
        render_text('Some random text', 20, 50, 1.0, (255, 100, 100))
        render_text('Some more random text', 120, 150, 1.0, (100, 255, 100))

        # Swap buffers
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == '__main__':
    main()