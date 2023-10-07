import glfw
import glfw.GLFW as GLFW_VAR
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader


'''
O------------------------------------------------------------------------------O
| GLFW CLOCK CLASS                                                             |
O------------------------------------------------------------------------------O
'''

class ClockGLFW:
    def __init__(self):
        self._fps_previous_time = glfw.get_time()
        self._fps_current_time = 0.0
        self._fps_frame_rate = 1
        self._num_frames = 0

        self._ft_previous_time = self._fps_previous_time
        self._ft_current_time = 0.0
        self._ft_frame_time = 1e-6

    def Update(self):
        # Update FPS counter
        self._fps_current_time = glfw.get_time()
        delta_time = self._fps_current_time - self._fps_previous_time
        if (delta_time >= 0.2):
            self._fps_frame_rate = int(self._num_frames / delta_time)
            self._fps_previous_time = self._fps_current_time
            self._num_frames = -1
        self._num_frames += 1
        # Update frame time duration
        self._ft_current_time = self._fps_current_time
        self._ft_frame_time = self._fps_current_time - self._ft_previous_time
        self._ft_previous_time = self._ft_current_time

    def ShowFrameRate(self, window):
        glfw.set_window_title(window, f'Frame rate : {self._fps_frame_rate} FPS')

    def ShowFrameTime(self, window):
        glfw.set_window_title(window, f'Frame time : {self._ft_frame_time:.6f} s')



'''
O------------------------------------------------------------------------------O
| MAIN OPENGL APP CLASS                                                        |
O------------------------------------------------------------------------------O
'''

# TODO : Render to custom size framebuffer first and then add to windowed quad.
# TODO : Reformat the code from here on !!!
# TODO : Show information in window title
# TODO : Move fixed stuff outside of the main render loop to update one demand (e.g., pix_size...)
# TODO : Add an window icon


class FractalRenderingApp():

    def __init__(self, window_size=(800, 600), range_x=(-2.0, 1.0)):
        self.win_size = np.asarray(window_size).astype('int')
        self.range_x_default = np.asarray(range_x)

        # Create GLFW window and update the window size
        self.window = self.create_window_glfw(self.win_size)
        self.win_size = np.asarray(glfw.get_window_size(self.window)).astype('int')

        # Create GLFW clock
        self.clock = ClockGLFW()

        # Create a shader program
        self.shader_program = self.create_shader_program('shaders/vertex_shader.glsl',
                                                    'shaders/fragment_shader.glsl')

        # Toggle flags
        self.win_open = True
        self.win_minimized = False
        self.keyboard_up_key_hold = False
        self.keyboard_down_key_hold = False
        self.mouse_middle_button_hold = False


        # Initialize number of iteration variables
        self.num_iter = 200
        self.num_iter_min = 20
        self.num_iter_max = 500
        self.num_iter_step = 20

        # Initialize shift and scale variables
        self.scale_min = 0.5
        self.scale_max = 1.0e16
        self.scale_step = 0.01
        self.shift_default, self.scale_default = self.compute_shift_and_scale(self.range_x_default, (0.0, 0.0), self.win_size)
        self.shift = self.shift_default.copy()
        self.scale = self.scale_default

        # Initialize the mouse pointer variables
        self.mp_s = np.asarray([0, 0], dtype='int')
        self.mp_s_previous = self.mp_s.copy()


        # Set GLFW callback functions
        glfw.set_window_close_callback(self.window, self.callback_window_close)
        glfw.set_window_size_callback(self.window, self.callback_window_size)
        glfw.set_window_iconify_callback(self.window, self.callback_window_iconified)
        glfw.set_cursor_pos_callback(self.window, self.callback_cursor_position)
        glfw.set_mouse_button_callback(self.window, self.callback_mouse_button)
        glfw.set_scroll_callback(self.window, self.callback_mouse_scroll)
        glfw.set_key_callback(self.window, self.callback_keyboad_button)


        # # Texture coordinates to display the image
        # textured_quad_vertices = np.asarray([[-1, -1, 0, 0],
        #                                         [-1,  1, 0, 1],
        #                                         [ 1,  1, 1, 1],
        #                                         [-1, -1, 0, 0],
        #                                         [ 1,  1, 1, 1],
        #                                         [ 1, -1, 1, 0]], dtype='float32')
        #
        # # Define Vertex Buffer Object (VBO)
        # window_quad_buffer = glGenBuffers(1)
        # glBindBuffer(GL_ARRAY_BUFFER, window_quad_buffer)
        # glBufferData(GL_ARRAY_BUFFER, textured_quad_vertices.nbytes, textured_quad_vertices, GL_STATIC_DRAW)
        #
        # # Define Vertex Array Object (VAO)
        # window_quad_vao = glGenVertexArrays(1)
        # glBindVertexArray(window_quad_vao)
        # glBindBuffer(GL_ARRAY_BUFFER, window_quad_buffer)
        #
        # # Enable VAO attributes (layout of the VBO)
        # glEnableVertexAttribArray(0)
        # glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        # glEnableVertexAttribArray(1)
        # glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))




        # Texture coordinates to display the image
        textured_quad_vertices = np.asarray([[-1, -1, 0, 0],
                                                [-1,  1, 0, 1],
                                                [ 1,  1, 1, 1],
                                                [-1, -1, 0, 0],
                                                [ 1,  1, 1, 1],
                                                [ 1, -1, 1, 0]], dtype='float32')

        # Define Vertex Buffer Object (VBO)
        self.window_quad_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.window_quad_buffer)
        glBufferData(GL_ARRAY_BUFFER, textured_quad_vertices.nbytes, textured_quad_vertices, GL_STATIC_DRAW)

        # Define Vertex Array Object (VAO)
        self.window_quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self.window_quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.window_quad_buffer)

        # Enable VAO attributes (layout of the VBO)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        # glEnableVertexAttribArray(1)
        # glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))




        # Set triangle color buffer
        triangle_color_data = np.zeros((1, 4), dtype='float32')
        triangle_color_data[:, 0:3] = np.asarray([1.0, 0.0, 0.0], dtype='float32')

        # Create SSBO
        self.pixel_centers_buffer = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.pixel_centers_buffer)
        glBufferData(GL_SHADER_STORAGE_BUFFER, triangle_color_data.nbytes, triangle_color_data, GL_STATIC_DRAW)

        # Uniform Buffer Object (UBO)
        triangle_data = np.asarray([0.8, 0.0, 0.0, 0.0,
                                      6.7, 0.0, 0.0, 0.0,
                                      1.0, 0.0, 0.0, 0.0], dtype='float32')
        self.ubo = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.ubo)
        glBufferData(GL_UNIFORM_BUFFER, triangle_data.nbytes, triangle_data, GL_STATIC_DRAW)

        # Get uniform locations
        self.uniform_locations = self.get_uniform_locations(self.shader_program)


        # Loop until the user closes the window
        while self.win_open:

            # Event handling
            glfw.poll_events()
            self.process_events()

            # Draw call
            if not self.win_minimized:
                self.draw_call()



        # Delete OpenGL bufferse
        glDeleteBuffers(1, [self.window_quad_buffer, self.pixel_centers_buffer])
        glDeleteVertexArrays(1, [self.window_quad_vao])
        glDeleteProgram(self.shader_program)
        # Terminate GLFW
        glfw.destroy_window(self.window)
        glfw.terminate()



    def draw_call(self):

        # Clear the screen
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Use the shader program
        glUseProgram(self.shader_program)
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.ubo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.pixel_centers_buffer)

        # Get ranges and pixel size
        range_x, range_y = self.get_window_range()
        pix_size = self.get_pixel_size(range_x, range_y, self.win_size)

        # Send uniforms to the GPU
        glUniform2dv(self.uniform_locations['pix_size'], 1, pix_size.astype('float64'))
        glUniform2dv(self.uniform_locations['range_x'], 1, range_x.astype('float64'))
        glUniform2dv(self.uniform_locations['range_y'], 1, range_y.astype('float64'))
        glUniform1i(self.uniform_locations['max_iter'], self.num_iter)

        # Draw the triangle
        glBindVertexArray(self.window_quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Update the clock
        self.clock.Update()
        self.clock.ShowFrameRate(self.window)

        # Swap buffers and poll for events
        glfw.swap_buffers(self.window)
        # glFlush()


    def compute_shift_and_scale(self, range_x, range_y, win_size):

        # Compute the scaling factor
        size_x = range_x[1] - range_x[0]
        size_y = range_y[1] - range_y[0]
        pix_size = size_x / win_size[0]
        scale = 1.0 / pix_size

        # Compute the shift
        temp_shift_x = 0.5 * win_size[0]  # Offset by image center
        temp_shift_x -= (range_x[0] + 0.5 * size_x) * scale  # Offset by x-extent
        temp_shift_y = -0.5 * win_size[1]  # Offset by image center
        temp_shift_y += (range_y[0] + 0.5 * size_y) * scale  # Offset by y-extent
        shift = np.asarray([temp_shift_x, temp_shift_y], dtype='float')

        # Return results
        return shift, scale


    def get_window_range(self):
        TL_w = self.s2w(np.asarray([0.0, 0.0]))
        BR_W = self.s2w(np.asarray(self.win_size))
        range_x = np.asarray([TL_w[0], BR_W[0]])
        range_y = np.asarray([BR_W[1], TL_w[1]])
        return range_x, range_y

    def get_pixel_size(self, range_x, range_y, win_size):
        pix_size = np.empty(shape=2, dtype='float')
        pix_size[0] = (range_x[1] - range_x[0]) / win_size[0]
        pix_size[1] = (range_y[1] - range_y[0]) / win_size[1]
        return pix_size


    # O------------------------------------------------------------------------------O
    # | SCREEN-TO-WORLD & WORLD-TO-SCREEN TRANSFORMATIONS                            |
    # O------------------------------------------------------------------------------O

    def s2w(self, points):
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = (points[0] - self.shift[0]) / self.scale
        output_points[1] = (self.win_size[1] + self.shift[1] - points[1]) / self.scale
        return output_points

    def w2s(self, points):
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = self.shift[0] + points[0] * self.scale
        output_points[1] = self.win_size[1] + self.shift[1] - points[1] * self.scale
        return output_points


    # O------------------------------------------------------------------------------O
    # | EVENT HANDLING - GLFW CALLBACK FUNCTIONS                                     |
    # O------------------------------------------------------------------------------O

    def callback_window_close(self, window):
        self.win_open = False


    def callback_window_iconified(self, window, iconified):
        self.win_minimized = bool(iconified)
        print(f'Window minimized = {self.win_minimized}')


    def callback_window_size(self, window, width, height):
        if not self.win_minimized:
            self.window_resize()
            self.draw_call()


    def callback_keyboad_button(self, window, key, scancode, action, mods):
        # Quit the app
        if (key == glfw.KEY_ESCAPE and action == glfw.PRESS):
            self.win_open = False
        # Increase number of iterations
        if (key == glfw.KEY_KP_ADD and action == glfw.PRESS):
            self.iterations_increase(self.num_iter_step)
        # Decrease number of iterations
        if (key == glfw.KEY_KP_SUBTRACT and action == glfw.PRESS):
            self.iterations_decrease(self.num_iter_step)
        # Reset shift and scale
        if (key == glfw.KEY_R and action == glfw.PRESS):
            self.window_default_shift_and_scale()
        # Hold zoom-in
        if (key == glfw.KEY_UP):
            if (action == glfw.PRESS):
                self.keyboard_up_key_hold = True
            elif (action == glfw.RELEASE):
                self.keyboard_up_key_hold = False
        # Hold zoom-out
        if (key == glfw.KEY_DOWN):
            if (action == glfw.PRESS):
                self.keyboard_down_key_hold = True
            elif (action == glfw.RELEASE):
                self.keyboard_down_key_hold = False


    def callback_mouse_button(self, window, button, action, mod):
        # Hold down the mouse button
        if (button == glfw.MOUSE_BUTTON_LEFT):
            if (action == glfw.PRESS):
                self.mouse_middle_button_hold = True
            elif (action == glfw.RELEASE):
                self.mouse_middle_button_hold = False



    def callback_mouse_scroll(self, window, x_offset, y_offset):
        # Zoom-in
        if (int(y_offset) == 1):
            self.window_scale_increase(self.scale_step)
        # Zoom-out
        if (int(y_offset) == -1):
            self.window_scale_decrease(self.scale_step)


    def callback_cursor_position(self, window, x_pos, y_pos):
        self.mp_s[0] = int(x_pos)
        self.mp_s[1] = int(y_pos)
        # DEBUG
        # print(f'Mouse cursor = {self.s2w(self.mp_s)}')

    def hold_key(self, key, action, glfw_key_identifier):
        if (key == glfw_key_identifier):
            if (action == glfw.PRESS):
                return True
            elif (action == glfw.RELEASE):
                return False

    # O------------------------------------------------------------------------------O
    # | EVENT HANDLING - UPDATE FUNCTIONS                                            |
    # O------------------------------------------------------------------------------O

    def process_events(self):
        # Drag the mouse
        if self.mouse_middle_button_hold:
            self.window_shift()
        # Zoom-in
        if self.keyboard_up_key_hold:
            self.window_scale_increase(self.scale_step)
        # Zoom-out
        if self.keyboard_down_key_hold:
            self.window_scale_decrease(self.scale_step)
        # Update previous mouse pointer position
        self.mp_s_previous = self.mp_s.copy()


    def window_resize(self):
        # Update App variables
        range_x, range_y = self.get_window_range()
        self.win_size = np.asarray(glfw.get_framebuffer_size(self.window)).astype('int')
        self.shift_default, self.scale_default = self.compute_shift_and_scale(self.range_x_default, (0.0, 0.0), self.win_size)
        self.shift, self.scale = self.compute_shift_and_scale(range_x, range_y, self.win_size)
        # Update OpenGL variables
        glViewport(0, 0, self.win_size[0], self.win_size[1])
        # DEBUG
        print(f'Window size = {self.win_size}')


    def window_shift(self):
        delta_shift = self.mp_s - self.mp_s_previous
        if (delta_shift[0] != 0.0) or (delta_shift[1] != 0.0):
            self.shift += delta_shift
            # DEBUG
            # print(f'Window shift = {self.shift}')


    def window_scale_increase(self, scale_step):
        temp_MP_w_start = self.s2w(self.mp_s)  # Starting position for the mouse
        self.scale *= (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self.scale / self.scale_default) > self.scale_max:
            self.scale = self.scale_max * self.scale_default  # Max zoom
        self.shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(temp_MP_w_start)  # Correct position by panning
        # DEBUG
        print(f'Window scale = {self.scale}')


    def window_scale_decrease(self, scale_step):
        temp_MP_w_start = self.s2w(self.mp_s)  # Starting position for the mouse
        self.scale *= 1.0 / (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self.scale / self.scale_default) < self.scale_min:
            self.scale = self.scale_min * self.scale_default  # Min zoom
        self.shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(temp_MP_w_start)  # Correct position by panning
        # DEBUG
        print(f'Window scale = {self.scale}')


    def window_default_shift_and_scale(self):
        self.shift = self.shift_default.copy()
        self.scale = self.scale_default


    def iterations_increase(self, num_iter_step):
        self.num_iter += num_iter_step
        if self.num_iter > self.num_iter_max:
            self.num_iter = self.num_iter_max
        # DEBUG
        print(f'Iterations = {self.num_iter}')


    def iterations_decrease(self, num_iter_step):
        self.num_iter -= num_iter_step
        if self.num_iter < self.num_iter_min:
            self.num_iter = self.num_iter_min
        # DEBUG
        print(f'Iterations = {self.num_iter}')


    # O------------------------------------------------------------------------------O
    # | OPENGL FUNCTIONS                                                             |
    # O------------------------------------------------------------------------------O

    def create_window_glfw(self, window_size):
        window_size = np.asarray(window_size)

        # Initialize GLFW
        glfw.init()
        glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MINOR, 4)
        glfw.window_hint(GLFW_VAR.GLFW_OPENGL_PROFILE, GLFW_VAR.GLFW_OPENGL_CORE_PROFILE)
        glfw.window_hint(GLFW_VAR.GLFW_OPENGL_FORWARD_COMPAT, GLFW_VAR.GLFW_TRUE)
        glfw.window_hint(GLFW_VAR.GLFW_SCALE_TO_MONITOR, GLFW_VAR.GLFW_TRUE)
        glfw.window_hint(GLFW_VAR.GLFW_DOUBLEBUFFER, GLFW_VAR.GLFW_TRUE)
        glfw.window_hint(GLFW_VAR.GLFW_RESIZABLE, GLFW_VAR.GLFW_TRUE)
        glfw.window_hint(GLFW_VAR.GLFW_FOCUSED, GLFW_VAR.GLFW_TRUE)

        # Create a GLFW window
        window = glfw.create_window(window_size[0], window_size[1], "Fractal Rendering", None, None)
        glfw.set_window_size_limits(window, 200, 200, GLFW_VAR.GLFW_DONT_CARE, GLFW_VAR.GLFW_DONT_CARE)
        glfw.make_context_current(window)
        glfw.swap_interval(1)  # V-sync
        return window


    def create_shader_program(self, vertex_shader_path, fragment_shader_path):

        # Open and load individual shaders
        with open(vertex_shader_path, 'r') as f:
            vertex_src = f.readlines()
        with open(fragment_shader_path, 'r') as f:
            fragment_src = f.readlines()

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



    def get_uniform_locations(self, shader_program):
        uniform_names = ['pix_size', 'range_x', 'range_y', 'max_iter']
        uniform_locations = {}
        for uniform_name in uniform_names:
            uniform_locations[uniform_name] = glGetUniformLocation(shader_program, uniform_name)
        return uniform_locations


# Main function call
if __name__ == '__main__':
    app = FractalRenderingApp()

