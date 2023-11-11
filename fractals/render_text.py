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

# TODO : Change the code such that text only gets added to a queue when calling "add_text" and gets renders all together
#      : when calling "draw_text".

class CharacterSlot:

    def __init__(self, ascii_id, glyph):
        self.ascii_id = ascii_id                              # ID of the ASCII character
        self.size = (glyph.bitmap.width, glyph.bitmap.rows)   # Size of glyph
        self.bearing = (glyph.bitmap_left, glyph.bitmap_top)  # Offset from the baseline to left/top of glyph
        self.advance = glyph.advance.x                        # Offset to advance to next glyph


class TextRenderer:

    def __init__(self):

        # Get max uniform block size
        self.max_instances = int(glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE) / (4 * 4 * 4))

        # Read shader source code
        shaders_path = os.path.join(os.path.abspath(__file__), os.pardir, 'shaders')
        vertex_shader_source = read_shader_source(os.path.join(shaders_path, 'text_render.vert'))
        fragment_shader_source = read_shader_source(os.path.join(shaders_path, 'text_render.frag'))
        # Dynamically modify the shader source code
        vertex_shader_source = vertex_shader_source.replace('INSERT_NUM_INSTANCES', str(self.max_instances))
        fragment_shader_source = fragment_shader_source.replace('INSERT_NUM_INSTANCES', str(self.max_instances))
        # Create a shader program
        self.shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
        glUseProgram(self.shader_program)

        # Enable text blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Get uniform locations
        self.uniform_locations = get_uniform_locations( self.shader_program, ['proj_mat'])

        # Set buffers
        self._set_vertex_buffer()
        self._set_trans_mat_buffer()
        self._set_char_id_buffer()
        self._set_color_buffer()

    def SetWindowSize(self, size):
        self.window_size = np.asarray(size).astype('int')
        self.proj_mat = np.array(glm.ortho(0, self.window_size[0], 0, self.window_size[1], -1, 1)).T


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
        # Bind the buffer
        trans_mat_buffer_block_index = glGetUniformBlockIndex(self.shader_program, 'trans_mat_buffer')
        glUniformBlockBinding(self.shader_program, trans_mat_buffer_block_index, 0)
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.trans_mat_buffer)


    def _set_char_id_buffer(self):
        # Initialize the data
        self.char_id_array = np.zeros(shape=(self.max_instances, 4), dtype='int32')
        # Set character ids buffer
        self.char_id_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.char_id_buffer)
        glBufferData(GL_UNIFORM_BUFFER, self.char_id_array.nbytes, None, GL_DYNAMIC_DRAW)
        # Bind the buffer
        char_id_buffer_block_index = glGetUniformBlockIndex(self.shader_program, 'char_id_buffer')
        glUniformBlockBinding(self.shader_program, char_id_buffer_block_index, 1)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.char_id_buffer)


    def _set_color_buffer(self):
        # Initialize the data
        self.color_array = np.zeros(shape=(self.max_instances, 4), dtype='float32')
        # Set color buffer
        self.color_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.color_buffer)
        glBufferData(GL_UNIFORM_BUFFER, self.color_array.nbytes, None, GL_DYNAMIC_DRAW)
        # Bind the buffer
        color_buffer_block_index = glGetUniformBlockIndex(self.shader_program, 'color_buffer')
        glUniformBlockBinding(self.shader_program, color_buffer_block_index, 2)
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, self.color_buffer)


    def Terminate(self):
        # Delete OpenGL buffers
        glDeleteBuffers(4, [self.texture_vbo, self.trans_mat_buffer, self.char_id_buffer, self.color_buffer])
        glDeleteVertexArrays(1, [self.texture_vao])
        glDeleteTextures(1, [self.texture_array])
        glDeleteProgram(self.shader_program)


    def SetFont(self, font_type, font_size):
        self.font_size = int(font_size)

        # Number of ASCII characters
        num_ASCII_char = 256

        # Load freetype characters
        face = freetype.Face(font_type)
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


    def AddText(self, text, x, y, scale, color):

        # Rescale the color to [0, 1] range
        text_color = (np.asarray(color) / 255.0).astype('float32')

        # Activate text rendering
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_array)
        glBindVertexArray(self.texture_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)

        # Save original x and y
        x_start = x

        # Flip y-axis for top-left origin
        y = self.window_size[1] - y - self.font_size * scale

        # Loop over all characters in the text
        index = 0
        for c in text:

            # Get the current character
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
                # Get dimensions
                x_pos = x + ch.bearing[0] * scale
                y_pos = y - (self.font_size - ch.bearing[1]) * scale

                # Create transformation matrix
                temp_trans_mat = (glm.translate(glm.mat4(1.0), glm.vec3(x_pos, y_pos, 0.0)) *
                                  glm.scale(glm.mat4(1.0), glm.vec3(self.font_size * scale, self.font_size * scale, 0.0)))

                # Set up the texture data
                self.trans_mat_array[index, :, :] = np.array(temp_trans_mat).T
                self.char_id_array[index, 0] = ch.ascii_id
                self.color_array[index, 0:3] = text_color

                # Advance cursors for next glyph
                x += (ch.advance >> 6) * scale

                # Update the working index
                index += 1

                # Draw call
                if (index == self.max_instances):
                    self.render(index)
                    index = 0

        # Final draw call
        self.render(index)


    def render(self, num_instances):

        # Send uniforms to the GPu
        glUniformMatrix4fv(self.uniform_locations['proj_mat'], 1, GL_FALSE, self.proj_mat)

        # Update transformation buffer
        temp_num_bytes = num_instances * 64  # 64 -> number of bytes of mat4
        temp_data = self.trans_mat_array[0: num_instances, :, :]
        glBindBuffer(GL_UNIFORM_BUFFER, self.trans_mat_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, temp_num_bytes, temp_data)

        # Update character ID buffer
        temp_num_bytes = num_instances * 16  # 16 -> number of bytes of int aligned to vec4
        temp_data = self.char_id_array[0: num_instances, :]
        glBindBuffer(GL_UNIFORM_BUFFER, self.char_id_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, temp_num_bytes, temp_data)

        # Update color buffer
        temp_num_bytes = num_instances * 16  # 16 -> number of bytes of vec4
        temp_data = self.color_array[0: num_instances, :]
        glBindBuffer(GL_UNIFORM_BUFFER, self.color_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, temp_num_bytes, temp_data)

        # Draw instanced characters
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
    text_renderer = TextRenderer()
    text_renderer.SetFont(r'C:\Windows\Fonts\arial.ttf', font_size=50)
    text_renderer.SetWindowSize(window_size)


    # Define some text
    test_text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit,\n' \
                'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n' \
                'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris\n' \
                'nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in\n' \
                'reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla\n' \
                'pariatur. Excepteur sint occaecat cupidatat non proident, sunt in\n' \
                'culpa qui officia deserunt mollit anim id est laborum.'

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
        text_renderer.AddText(test_text, 0, 0, 0.5, (255, 0, 0))
        text_renderer.AddText(test_text, 1, 400, 1.0, (0, 255, 0))

        # Swap buffers
        glfw.swap_buffers(window)

    # Terminate the app
    text_renderer.Terminate()
    glfw.terminate()


# Run main function
if __name__ == '__main__':
    main()