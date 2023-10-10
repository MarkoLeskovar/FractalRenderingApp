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
| CANVAS CLASS FOR WINDOW DRAWING AREA HANDLING                                |
O------------------------------------------------------------------------------O
'''

# TODO : Update variables only when needed and not on every iteration
# TODO : Add "S2GL" and "GL2S" transformation functions to switch between Screen and OpenGL
# TODO : Experiment with customizable drawing area via convex polygon triangulation

class Canvas():

    def __init__(self, size=(400, 300), range_x=(-1, 1)):
        self.size = np.asarray(size).astype('int')
        self.range_x_default = np.asarray(range_x).astype('float')

        # Scaling settings
        self.scale_rel_min = 0.5
        self.scale_rel_max = 1.0e15
        self.scale_abs_step = 0.02

        # Default shift and scale variables
        self.shift_default, self.scale_abs_default = self.GetShiftAndScale(self.range_x_default, (0.0, 0.0))
        self.shift = self.shift_default.copy()
        self.scale_abs = self.scale_abs_default

        # Initialize mouse position
        self.mouse_pos = np.asarray([0, 0], dtype='int')
        self.mouse_pos_previous = self.mouse_pos.copy()


    def GetShiftAndScale(self, range_x, range_y):
        # Compute the scaling factor
        size_x = range_x[1] - range_x[0]
        size_y = range_y[1] - range_y[0]
        pix_size = size_x / self.size[0]
        scale = 1.0 / pix_size
        # Compute the shift
        temp_shift_x = 0.5 * self.size[0]  # Offset by image center
        temp_shift_x -= (range_x[0] + 0.5 * size_x) * scale  # Offset by x-extent
        temp_shift_y = -0.5 * self.size[1]  # Offset by image center
        temp_shift_y += (range_y[0] + 0.5 * size_y) * scale  # Offset by y-extent
        shift = np.asarray([temp_shift_x, temp_shift_y], dtype='float')
        # Return results
        return shift, scale


    def GetRangeXY(self):
        TL_w = self.S2W(np.asarray([0.0, 0.0]))
        BR_W = self.S2W(np.asarray(self.size))
        range_x = np.asarray([TL_w[0], BR_W[0]])
        range_y = np.asarray([BR_W[1], TL_w[1]])
        return range_x, range_y


    def GetPixelSize(self, range_x):
        return (range_x[1] - range_x[0]) / self.size[0]


    def ResetShiftAndScale(self):
        self.shift = self.shift_default.copy()
        self.scale_abs = self.scale_abs_default


    def Resize(self, size):
        range_x, range_y = self.GetRangeXY()
        self.size = np.asarray(size).astype('int')
        self.shift_default, self.scale_abs_default = self.GetShiftAndScale(self.range_x_default, (0.0, 0.0))
        self.shift, self.scale_abs = self.GetShiftAndScale(range_x, range_y)


    def UpdateShift(self):
        delta_shift = self.mouse_pos - self.mouse_pos_previous
        self.shift += delta_shift


    def ScaleIncrease(self, scale_step):
        temp_MP_w_start = self.S2W(self.mouse_pos)  # Starting position for the mouse
        self.scale_abs *= (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self.scale_abs / self.scale_abs_default) > self.scale_rel_max:
            self.scale_abs = self.scale_rel_max * self.scale_abs_default  # Max zoom
        self.shift += self.W2S(self.S2W(self.mouse_pos)) - self.W2S(temp_MP_w_start)  # Correct position by panning
        # DEBUG
        print(f'Window scale = {self.scale_abs / self.scale_abs_default:.2e}')


    def ScaleDecrease(self, scale_step):
        temp_MP_w_start = self.S2W(self.mouse_pos)  # Starting position for the mouse
        self.scale_abs /= (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self.scale_abs / self.scale_abs_default) < self.scale_rel_min:
            self.scale_abs = self.scale_rel_min * self.scale_abs_default  # Min zoom
        self.shift += self.W2S(self.S2W(self.mouse_pos)) - self.W2S(temp_MP_w_start)  # Correct position by panning
        # DEBUG
        print(f'Window scale = {self.scale_abs / self.scale_abs_default:.2e}')


    def SetMousePos(self, pos):
        self.mouse_pos = np.asarray(pos).astype('int')


    def UpdateMousePosPrevious(self):
        self.mouse_pos_previous = self.mouse_pos.copy()


    # O------------------------------------------------------------------------------O
    # | SCREEN-TO-WORLD & WORLD-TO-SCREEN TRANSFORMATIONS                            |
    # O------------------------------------------------------------------------------O

    def S2W(self, points):
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = (points[0] - self.shift[0]) / self.scale_abs
        output_points[1] = (self.size[1] + self.shift[1] - points[1]) / self.scale_abs
        return output_points

    def W2S(self, points):
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = self.shift[0] + points[0] * self.scale_abs
        output_points[1] = self.size[1] + self.shift[1] - points[1] * self.scale_abs
        return output_points



'''
O------------------------------------------------------------------------------O
| MAIN OPENGL APP CLASS                                                        |
O------------------------------------------------------------------------------O
'''

# TODO : Add icon to window
# TODO : Add option to toggle full-screen window with "F". This must be adapted to each monitor
# TODO : Add functionality to save a screenshot of actual render to a file
# TODO : Check if OpenGL functions are implemented correctly
# TODO : Move windowed quad to a separate function
# TODO : Add text rendering to display information on the screen
# TODO : Refactor the code to seperate render passes


class FractalRenderingApp():

    def __init__(self, window_size=(800, 600), range_x=(-2.0, 1.0)):

        # Create GLFW window and pixel size
        self.window = self.create_window_glfw(window_size)
        self.pix_scale = float(glfw.get_window_content_scale(self.window)[0])

        # Get the actual window size
        self.window_size = np.asarray(glfw.get_framebuffer_size(self.window)).astype('int')
        self.render_size = (self.window_size / self.pix_scale).astype('int')

        # Create a canvas
        self.canvas = Canvas(self.render_size, range_x)

        # Create GLFW clock
        self.clock = ClockGLFW()

        # Set GLFW callback functions
        self.set_callback_functions_glfw()

        # Fractal interation variables
        self.num_iter = 256
        self.num_iter_min = 32
        self.num_iter_max = 1024
        self.num_iter_step = 32

        # Pixel scaling variables
        self.pix_scale_min = 0.5
        self.pix_scale_max = 8.0
        self.pix_scale_step = 0.25

        # Main shader program
        self.program_main = self.create_shader_program('shaders/vertex_shader.glsl',
                                                       'shaders/fragment_shader_main.glsl')
        # Post-processing shader program
        self.program_post = self.create_shader_program('shaders/vertex_shader.glsl',
                                                       'shaders/fragment_shader_post.glsl')

        # Get uniform locations
        self.uniform_locations_main = self.get_uniform_locations(self.program_main)

        # Create framebuffers
        self.framebuffer_main, self.texture_main = self.create_framebuffer(self.render_size, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE)
        self.framebuffer_post, self.texture_post = self.create_framebuffer(self.window_size, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE)




        # Texture coordinates to display the image
        textured_quad_vertices = np.asarray([[-1, -1, 0, 0],
                                                [-1,  1, 0, 1],
                                                [ 1,  1, 1, 1],
                                                [-1, -1, 0, 0],
                                                [ 1,  1, 1, 1],
                                                [ 1, -1, 1, 0]], dtype='float32')

        # Define Vertex Buffer Object (VBO)
        self.quad_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_buffer)
        glBufferData(GL_ARRAY_BUFFER, textured_quad_vertices.nbytes, textured_quad_vertices, GL_STATIC_DRAW)

        # Define Vertex Array Object (VAO)
        self.quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_buffer)

        # Enable VAO attributes (layout of the VBO)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))


        # Loop until the user closes the window
        while self.window_open:

            # Draw call
            if not self.window_minimized:
                self.draw_call()

            # Event handling
            glfw.poll_events()
            self.process_hold_keys()
            self.canvas.UpdateMousePosPrevious()


        # TODO : Check if everything is deleted
        # Delete OpenGL buffers
        glDeleteBuffers(1, [self.quad_buffer])
        glDeleteVertexArrays(1, [self.quad_vao])
        glDeleteFramebuffers(1, [self.framebuffer_main, self.framebuffer_post])
        glDeleteTextures(1, [self.texture_main, self.texture_post])
        glDeleteProgram(self.program_main)
        # Terminate GLFW
        glfw.destroy_window(self.window)
        glfw.terminate()


    # O------------------------------------------------------------------------------O
    # | GLFW EVENT HANDLING - CALLBACK FUNCTIONS AND USER INPUT                      |
    # O------------------------------------------------------------------------------O

    def set_callback_functions_glfw(self):
        # Toggle flags
        self.window_open = True
        self.window_minimized = False
        self.keyboard_up_key_hold = False
        self.keyboard_down_key_hold = False
        self.mouse_left_button_hold = False
        # Window callback functions
        glfw.set_window_close_callback(self.window, self.callback_window_close)
        glfw.set_window_size_callback(self.window, self.callback_window_resize)
        glfw.set_window_iconify_callback(self.window, self.callback_window_minimized)
        glfw.set_window_content_scale_callback(self.window, self.callback_content_scale)
        # User input callback functions
        glfw.set_cursor_pos_callback(self.window, self.callback_cursor_position)
        glfw.set_mouse_button_callback(self.window, self.callback_mouse_button)
        glfw.set_scroll_callback(self.window, self.callback_mouse_scroll)
        glfw.set_key_callback(self.window, self.callback_keyboad_button)


    def callback_window_close(self, window):
        self.window_open = False


    def callback_window_minimized(self, window, iconified):
        self.window_minimized = bool(iconified)


    def callback_window_resize(self, window, width, height):
        if not self.window_minimized:
            temp_size = glfw.get_framebuffer_size(self.window)
            self.window_size_update(temp_size, self.pix_scale)
            self.draw_call()


    def callback_content_scale(self, window, scale_x, scale_y):
        self.window_size_update(self.window_size, scale_x)


    def callback_keyboad_button(self, window, key, scancode, action, mods):
        # Quit the app
        if (key == glfw.KEY_ESCAPE and action == glfw.PRESS):
            self.window_open = False

        # Increase number of iterations
        if (key == glfw.KEY_KP_ADD and action == glfw.PRESS):
            self.num_iter = min(self.num_iter + self.num_iter_step, self.num_iter_max)
            # DEBUG
            print(f'Fractal iterations = {self.num_iter}')


        # Decrease number of iterations
        if (key == glfw.KEY_KP_SUBTRACT and action == glfw.PRESS):
            self.num_iter = max(self.num_iter - self.num_iter_step, self.num_iter_min)
            # DEBUG
            print(f'Fractal iterations = {self.num_iter}')


        # Reset shift and scale
        if (key == glfw.KEY_R and action == glfw.PRESS):
            self.canvas.ResetShiftAndScale()


        # Increase pixel scale
        if (key == glfw.KEY_KP_DIVIDE and action == glfw.PRESS):
            temp_pix_scale = min(self.pix_scale + self.pix_scale_step, self.pix_scale_max)
            self.window_size_update(self.window_size, temp_pix_scale)


        # Decrease pixel scale
        if (key == glfw.KEY_KP_MULTIPLY and action == glfw.PRESS):
            temp_pix_scale = max(self.pix_scale - self.pix_scale_step, self.pix_scale_min)
            self.window_size_update(self.window_size, temp_pix_scale)


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
                self.mouse_left_button_hold = True
            elif (action == glfw.RELEASE):
                self.mouse_left_button_hold = False


    def callback_mouse_scroll(self, window, x_offset, y_offset):
        # Zoom-in
        if (int(y_offset) == 1):
            self.canvas.ScaleIncrease(5.0 * self.canvas.scale_abs_step)
        # Zoom-out
        if (int(y_offset) == -1):
            self.canvas.ScaleDecrease(5.0 * self.canvas.scale_abs_step)


    def callback_cursor_position(self, window, x_pos, y_pos):
        temp_mp_s = np.asarray([x_pos, y_pos]) / self.pix_scale
        temp_mp_s = temp_mp_s.astype('int')
        self.canvas.SetMousePos(temp_mp_s)
        # DEBUG
        # print(f'Mouse cursor = {self.canvas.S2W(self.canvas.mouse_pos)}')


    def process_hold_keys(self):
        # Pan screen
        if self.mouse_left_button_hold:
            self.canvas.UpdateShift()
        # Zoom-in
        if self.keyboard_up_key_hold:
            self.canvas.ScaleIncrease(self.canvas.scale_abs_step)
        # Zoom-out
        if self.keyboard_down_key_hold:
            self.canvas.ScaleDecrease(self.canvas.scale_abs_step)


    def window_size_update(self, size, pix_scale):
        self.window_size = np.asarray(size).astype('int')
        self.pix_scale = float(pix_scale)
        # Update app variables
        self.render_size = (self.window_size / self.pix_scale).astype('int')
        self.canvas.Resize(self.render_size)
        # Update OpenGL framebuffers
        glDeleteFramebuffers(1, [self.framebuffer_main, self.framebuffer_post])
        glDeleteTextures(1, [self.texture_main, self.texture_post])
        self.framebuffer_main, self.texture_main = self.create_framebuffer(self.render_size, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE)
        self.framebuffer_post, self.texture_post = self.create_framebuffer(self.window_size, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE)
        # DEBUG
        print(f'Window size = {self.window_size}')
        print(f'Render size = {self.render_size}')
        print(f'Pixel scale = {self.pix_scale}')


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


    def create_framebuffer(self, size, gl_internalformat, gl_format, gl_type):
        # Create a frame-buffer object
        framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
        # Create a texture for a frame-buffer
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, gl_internalformat, size[0], size[1], 0, gl_format, gl_type,None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        # Return results
        return framebuffer, texture


    def draw_call(self):

        # 00. COMPUTE FRAME VARIABLES
        range_x, range_y = self.canvas.GetRangeXY()
        pix_size = self.canvas.GetPixelSize(range_x)

        # 01. MAIN RENDER PASS
        glViewport(0, 0, self.render_size[0], self.render_size[1])
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.framebuffer_main)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program_main)
        # Bind resources
        glBindVertexArray(self.quad_vao)
        # Send uniforms to the GPU
        glUniform2dv(self.uniform_locations_main['range_x'], 1, range_x.astype('float64'))
        glUniform2dv(self.uniform_locations_main['range_y'], 1, range_y.astype('float64'))
        glUniform1d(self.uniform_locations_main['pix_size'], pix_size)
        glUniform1i(self.uniform_locations_main['max_iter'], self.num_iter)
        # Draw geometry
        glDrawArrays(GL_TRIANGLES, 0, 6)


        # # DEBUG - READ PIXELS
        # image_screenshot = np.empty(shape=(self.render_size[0], self.render_size[1], 3), dtype='uint8')
        # glBindFramebuffer(GL_READ_FRAMEBUFFER, self.framebuffer_main)
        # glReadPixels(0, 0, self.render_size[0], self.render_size[1], GL_RGB, GL_UNSIGNED_BYTE, image_screenshot)
        # image_screenshot = image_screenshot.reshape((self.render_size[1], self.render_size[0], 3))


        # 02. POST-PROCESSING PASS
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.framebuffer_post)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program_post)
        # Bind resources
        glBindTexture(GL_TEXTURE_2D, self.texture_main)
        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self.quad_vao)
        # Draw geometry
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # 03. COPY FRAMEBUFFERS
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.framebuffer_post)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, self.window_size[0], self.window_size[1],
                          0, 0, self.window_size[0], self.window_size[1],
                          GL_COLOR_BUFFER_BIT, GL_NEAREST)

        # 04. SWAP BUFFERS
        glfw.swap_buffers(self.window)

        # 05. UPDATE THE TIMINGS
        self.clock.Update()
        self.clock.ShowFrameRate(self.window)


# Main function call
if __name__ == '__main__':
    app = FractalRenderingApp()

