import os
import tripy
import numpy as np
from PIL import Image

# OpenGL modules
import glfw
import glfw.GLFW as GLFW_VAR
from OpenGL.GL import *

# Add custom modules
from .canvas import Canvas
from .clock import ClockGLFW
from .text_render import TextRender
from .color import GetColormapArray, LoadColormapsFile
from .shader_utils import create_shader_program, read_shader_source, get_uniform_locations


# Default keymap dictionary
DEFAULT_SETTINGS = {
    # Keymap
    'EXIT':                 'KEY_ESCAPE',
    'INFO':                 'KEY_I',
    'VSYNC':                'KEY_V',
    'FULLSCREEN':           'KEY_F',
    'SCREENSHOT':           'KEY_S',
    'ZOOM_IN':              'KEY_UP',
    'ZOOM_OUT':             'KEY_DOWN',
    'SHIFT_VIEW':           'MOUSE_BUTTON_LEFT',
    'RESET_VIEW':           'KEY_R',
    'ITER_INCREASE':        'KEY_KP_ADD',
    'ITER_DECREASE':        'KEY_KP_SUBTRACT',
    'PIX_SCALE_INCREASE':   'KEY_KP_MULTIPLY',
    'PIX_SCALE_DECREASE':   'KEY_KP_DIVIDE',
    'CMAP_NEXT':            'KEY_RIGHT',
    'CMAP_PREV':            'KEY_LEFT',
    # Fractal iterations
    'NUM_ITER':             64,
    'NUM_ITER_MIN':         64,
    'NUM_ITER_MAX':         2048,
    'NUM_ITER_STEP':        32,
    # Pixel scaling
    'PIX_SCALE_MIN':        0.5,
    'PIX_SCALE_MAX':        8.0,
    'PIX_SCALE_STEP':       0.25,
    # On-screen text
    'TEXT_SIZE':            20,
}

# Default colormaps list
DEFAULT_CMAPS = [
    'cet_l_tritanopic_krjcw1_r',
    'cet_l_protanopic_deuteranopic_kbw_r',
    'jungle_r',
    'apple_r'
    ]

# Define default output directory
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), 'Pictures', 'FractalRendering')


'''
O------------------------------------------------------------------------------O
| MAIN OPENGL APP CLASS                                                        |
O------------------------------------------------------------------------------O
'''

# TODO : Refactor the code to make it nicer and easier to read
# TODO : Check if OpenGL functions are implemented correctly
# TODO : Add Julia Set rendering that changes depending on the position of the mouse pointer


class FractalRenderingApp:

    def __init__(self, window_size=(800, 600), range_x=(-2.0, 1.0), cmap_file=None, settings_file=None, output_dir=None):

        # Load colormaps file
        self.cmap_id = 0
        self.cmap_names = DEFAULT_CMAPS
        if cmap_file is not None:
            self.cmap_names = LoadColormapsFile(cmap_file)

        # Load settings file
        self.settings = DEFAULT_SETTINGS
        if settings_file is not None:
            self.settings = self.load_settings_file(settings_file)

        # Output directory for screenshots
        self.output_dir = DEFAULT_OUTPUT_DIR
        if output_dir is not None:
            self.output_dir = output_dir

        # Fractal interation settings
        self.num_iter = int(self.settings['NUM_ITER'])
        self.num_iter_min = int(self.settings['NUM_ITER_MIN'])
        self.num_iter_max = int(self.settings['NUM_ITER_MAX'])
        self.num_iter_step = int(self.settings['NUM_ITER_STEP'])

        # Pixel scaling settings
        self.pix_scale_min = float(self.settings['PIX_SCALE_MIN'])
        self.pix_scale_max = float(self.settings['PIX_SCALE_MAX'])
        self.pix_scale_step = float(self.settings['PIX_SCALE_STEP'])

        # Create GLFW window and set the icon
        self.window_vsync = True
        self.window = self.create_main_window(window_size, self.window_vsync)
        icon_path = os.path.join(os.path.abspath(__file__), os.pardir, 'assets', 'mandelbrot_set.png')
        icon = Image.open(icon_path).resize((32, 32))
        glfw.set_window_icon(self.window, 1, icon)

        # Get the actual window size
        self.pix_scale = float(glfw.get_window_content_scale(self.window)[0])
        self.window_size = np.asarray(glfw.get_framebuffer_size(self.window)).astype('int')
        self.render_size = (self.window_size / self.pix_scale).astype('int')

        # Create a canvas
        self.canvas = Canvas(self.render_size, range_x)

        # Create GLFW clock
        self.clock = ClockGLFW()

        # Create a text renderer
        self.text_size = int(self.settings['TEXT_SIZE'])
        self.text_font_type = os.path.join(os.path.abspath(__file__), os.pardir, 'assets', 'NotoMono-Regular.ttf')
        self.text_render = TextRender()
        self.text_render.SetFont(self.text_font_type, self.text_size * self.pix_scale)
        self.text_render.SetWindowSize(self.window_size)

        # Set GLFW callback functions
        self.set_callback_functions_glfw()

        # Read shader source code
        shaders_path = os.path.join(os.path.abspath(__file__), os.pardir, 'shaders')
        base_vert_source = read_shader_source(os.path.join(shaders_path, 'fractal_base.vert'))
        color_frag_source = read_shader_source(os.path.join(shaders_path, 'fractal_color.frag'))
        mandelbrot_frag_source = read_shader_source(os.path.join(shaders_path, 'fractal_mandelbrot.frag'))
        # Create shader programs
        self.program_mandelbrot = create_shader_program(base_vert_source, mandelbrot_frag_source)
        self.program_color = create_shader_program(base_vert_source, color_frag_source)


        # Get uniform locations
        self.uniform_locations_mandelbrot = get_uniform_locations(
            self.program_mandelbrot, ['pix_size', 'range_x', 'range_y', 'num_iter'])
        self.uniform_locations_color = get_uniform_locations(
            self.program_color, ['num_iter'])

        # Create framebuffers
        self.framebuffer_mandelbrot, self.texture_mandelbrot = self.create_framebuffer(self.render_size, GL_R32F, GL_RED, GL_FLOAT)
        self.framebuffer_color, self.texture_color = self.create_framebuffer(self.render_size, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE)


        # TODO : Create textured polygon class
        # Create vertices for textured polygon
        # points = np.asarray([[0, 0], [0, 600], [600, 600], [600, 400], [800, 400], [800, 0]])
        points = np.asarray([[0, 0], [0, self.canvas.size[1]], self.canvas.size, [self.canvas.size[0], 0]])
        textured_polygon_vertices = self.create_textured_polygon_vertices(points, self.canvas)
        self.num_polygon_vertices = textured_polygon_vertices.shape[0]

        # Polygon VBO and VAO
        self.polygon_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.polygon_buffer)
        glBufferData(GL_ARRAY_BUFFER, textured_polygon_vertices.nbytes, textured_polygon_vertices, GL_STATIC_DRAW)
        self.polygon_vao = glGenVertexArrays(1)
        glBindVertexArray(self.polygon_vao)
        # Enable VAO attributes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

        # Create buffers
        self.set_cmap_buffer(self.cmap_names[self.cmap_id])


    def Run(self):
        # Main app loop
        self.window_open = True
        while self.window_open:
            # Draw call
            if not self.window_minimized:
                self.draw_call()
            # Event handling
            glfw.poll_events()
            self.process_hold_keys()
            self.canvas.UpdateMousePosPrevious()


    def Close(self):
        # Delete text renderer
        self.text_render.Delete()
        # Delete OpenGL buffers
        glDeleteBuffers(2, [self.polygon_buffer, self.cmap_buffer])
        glDeleteVertexArrays(1, [self.polygon_vao])
        glDeleteFramebuffers(2, [self.framebuffer_mandelbrot, self.framebuffer_color])
        glDeleteTextures(2, [self.texture_mandelbrot, self.texture_color])
        glDeleteProgram(self.program_mandelbrot)
        glDeleteProgram(self.program_color)
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
        self.window_fullscreen = False
        self.keyboard_up_key_hold = False
        self.keyboard_down_key_hold = False
        self.mouse_left_button_hold = False
        self.show_info_text = True
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
            self.update_window_size(temp_size, self.pix_scale)
            self.text_render.SetWindowSize((width, height))
            self.draw_call()


    def callback_content_scale(self, window, scale_x, scale_y):
        self.update_window_size(self.window_size, scale_x)
        self.text_render.SetFont(self.text_font_type, self.text_size * self.pix_scale)
        self.draw_call()


    def callback_keyboad_button(self, window, key, scancode, action, mods):
        # Exit the app
        if (key == getattr(glfw, self.settings['EXIT']) and action == glfw.PRESS):
            self.window_open = False

        # Show into text
        if (key == getattr(glfw, self.settings['INFO']) and action == glfw.PRESS):
            self.show_info_text = not self.show_info_text

        # Toggle fullscreen
        if (key == getattr(glfw, self.settings['FULLSCREEN']) and action == glfw.PRESS):
            # Make fullscreen
            if not self.window_fullscreen:
                self.window_fullscreen = True
                self.window_pos_previous = np.asarray(glfw.get_window_pos(self.window)).astype('int')
                self.window_size_previous = self.window_size.copy()
                monitor = glfw_get_current_window_monitor(self.window)
                mode = glfw.get_video_mode(monitor)
                glfw.set_window_monitor(self.window, monitor, 0, 0, mode.size[0], mode.size[1], mode.refresh_rate)
                glfw.swap_interval(int(self.window_vsync))  # V-sync (refresh rate limit)
            # Make windowed
            else:
                self.window_fullscreen = False
                glfw.set_window_monitor(self.window, None, self.window_pos_previous[0], self.window_pos_previous[1],
                                        self.window_size_previous[0], self.window_size_previous[1], glfw.DONT_CARE)

        # Increase number of iterations
        if (key == getattr(glfw, self.settings['ITER_INCREASE']) and action == glfw.PRESS):
            self.num_iter = min(self.num_iter + self.num_iter_step, self.num_iter_max)


        # Decrease number of iterations
        if (key == getattr(glfw, self.settings['ITER_DECREASE']) and action == glfw.PRESS):
            self.num_iter = max(self.num_iter - self.num_iter_step, self.num_iter_min)


        # Reset shift and scale
        if (key == getattr(glfw, self.settings['RESET_VIEW']) and action == glfw.PRESS):
            self.canvas.ResetShiftAndScale()
            # Also reset the number of fractals iterations
            self.num_iter = self.num_iter_min

        # Increase pixel scale
        if (key == getattr(glfw, self.settings['PIX_SCALE_INCREASE']) and action == glfw.PRESS):
            temp_pix_scale = min(self.pix_scale + self.pix_scale_step, self.pix_scale_max)
            self.update_window_size(self.window_size, temp_pix_scale)

        # Decrease pixel scale
        if (key == getattr(glfw, self.settings['PIX_SCALE_DECREASE']) and action == glfw.PRESS):
            temp_pix_scale = max(self.pix_scale - self.pix_scale_step, self.pix_scale_min)
            self.update_window_size(self.window_size, temp_pix_scale)

        # Hold zoom-in
        if (key == getattr(glfw, self.settings['ZOOM_IN'])):
            if (action == glfw.PRESS):
                self.keyboard_up_key_hold = True
            elif (action == glfw.RELEASE):
                self.keyboard_up_key_hold = False

        # Hold zoom-out
        if (key == getattr(glfw, self.settings['ZOOM_OUT'])):
            if (action == glfw.PRESS):
                self.keyboard_down_key_hold = True
            elif (action == glfw.RELEASE):
                self.keyboard_down_key_hold = False

        # Next colormap
        if (key == getattr(glfw, self.settings['CMAP_NEXT']) and action == glfw.PRESS):
            self.cmap_id += 1
            if self.cmap_id >= len(self.cmap_names):
                self.cmap_id = 0
            self.update_cmap_buffer(self.cmap_names[self.cmap_id])

        # Previous colormap
        if (key == getattr(glfw, self.settings['CMAP_PREV']) and action == glfw.PRESS):
            self.cmap_id -= 1
            if self.cmap_id < 0:
                self.cmap_id = len(self.cmap_names) - 1
            self.update_cmap_buffer(self.cmap_names[self.cmap_id])

        # Toggle vsync for uncapped frame rate
        if (key == getattr(glfw, self.settings['VSYNC']) and action == glfw.PRESS):
            self.window_vsync = not self.window_vsync
            glfw.swap_interval(int(self.window_vsync))


        # TODO : Format a code in a nice way !!!
        # TODO : Add metadata information function (getFucntion)
        # Save a screenshot
        if (key == getattr(glfw, self.settings['SCREENSHOT']) and action == glfw.PRESS):
            os.makedirs(self.output_dir, exist_ok=True)
            # Get current screenshot counter
            counter = 1
            output_files = os.listdir(self.output_dir)
            if len(output_files) > 0:
                temp_list = sorted([int(x.split('.')[0]) for x in output_files])
                counter = temp_list[-1] + 1
            # Read pixels and save the image
            output_image = self.read_pixels(self.framebuffer_color, self.render_size)
            image_pil = Image.fromarray(np.flipud(output_image))
            image_pil.save(os.path.join(self.output_dir, f'{counter}.png'))


    def callback_mouse_button(self, window, button, action, mod):
        # Hold down the mouse button
        if (button == getattr(glfw, self.settings['SHIFT_VIEW'])):
            if (action == glfw.PRESS):
                self.mouse_left_button_hold = True
            elif (action == glfw.RELEASE):
                self.mouse_left_button_hold = False


    def callback_mouse_scroll(self, window, x_offset, y_offset):
        # Zoom-in
        if (y_offset > 0):
            temp_scale_step = 5.0 * self.canvas.scale_abs_step * abs(y_offset)
            self.canvas.ScaleIncrease(temp_scale_step)
        # Zoom-out
        if (y_offset < 0):
            temp_scale_step = 5.0 * self.canvas.scale_abs_step * abs(y_offset)
            self.canvas.ScaleDecrease(temp_scale_step)


    def callback_cursor_position(self, window, x_pos, y_pos):
        temp_mp_s = np.asarray([x_pos, y_pos]) / self.pix_scale
        self.canvas.SetMousePos(temp_mp_s)


    def process_hold_keys(self):
        # Pan screen
        if self.mouse_left_button_hold:
            self.canvas.UpdateShift()
        # Zoom-in
        if self.keyboard_up_key_hold:
            temp_scale_step = self.canvas.scale_abs_step * self.clock.frame_time * 60.0
            self.canvas.ScaleIncrease(temp_scale_step)
        # Zoom-out
        if self.keyboard_down_key_hold:
            temp_scale_step = self.canvas.scale_abs_step * self.clock.frame_time * 60.0
            self.canvas.ScaleDecrease(temp_scale_step)


    def update_window_size(self, size, pix_scale):
        # Update mouse position
        temp_mp_s = self.canvas.mouse_pos * (self.pix_scale / pix_scale)
        self.canvas.SetMousePos(temp_mp_s)
        # Update window size
        self.window_size = np.asarray(size).astype('int')
        self.pix_scale = float(pix_scale)
        # Update app variables
        self.render_size = (self.window_size / self.pix_scale).astype('int')
        self.canvas.Resize(self.render_size)
        # Update OpenGL framebuffers
        glDeleteFramebuffers(2, [self.framebuffer_mandelbrot, self.framebuffer_color])
        glDeleteTextures(2, [self.texture_mandelbrot, self.texture_color])
        self.framebuffer_mandelbrot, self.texture_mandelbrot = self.create_framebuffer(self.render_size, GL_R32F, GL_RED, GL_FLOAT)
        self.framebuffer_color, self.texture_color = self.create_framebuffer(self.render_size, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE)


    def load_settings_file(self, keymap_file):
        # Open the file
        with open(keymap_file, mode='r') as f:
            lines_raw = f.readlines()
        # Parse the lines
        keymap_dict = {}
        for line in lines_raw:
            line = line.split('#')[0]
            line = line.strip('\n').replace(' ', '')
            if len(line) != 0:
                line = line.split(':')
                keymap_dict[line[0]] = line[1]
        return keymap_dict


    # O------------------------------------------------------------------------------O
    # | OPENGL FUNCTIONS                                                             |
    # O------------------------------------------------------------------------------O

    def create_main_window(self, size, vsync=True):
        size = np.asarray(size).astype('int')
        # Initialize GLFW
        glfw.init()
        glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MINOR, 0)
        glfw.window_hint(GLFW_VAR.GLFW_OPENGL_PROFILE, GLFW_VAR.GLFW_OPENGL_CORE_PROFILE)
        glfw.window_hint(GLFW_VAR.GLFW_OPENGL_FORWARD_COMPAT, GLFW_VAR.GLFW_TRUE)
        glfw.window_hint(GLFW_VAR.GLFW_SCALE_TO_MONITOR, GLFW_VAR.GLFW_TRUE)
        glfw.window_hint(GLFW_VAR.GLFW_DOUBLEBUFFER, GLFW_VAR.GLFW_TRUE)
        glfw.window_hint(GLFW_VAR.GLFW_RESIZABLE, GLFW_VAR.GLFW_TRUE)
        glfw.window_hint(GLFW_VAR.GLFW_FOCUSED, GLFW_VAR.GLFW_TRUE)
        # Create a GLFW window
        window = glfw.create_window(size[0], size[1], "Fractal Rendering", None, None)
        glfw.set_window_size_limits(window, 200, 200, GLFW_VAR.GLFW_DONT_CARE, GLFW_VAR.GLFW_DONT_CARE)
        glfw.make_context_current(window)
        glfw.swap_interval(int(vsync))  # V-sync (refresh rate limit)
        return window


    def create_textured_polygon_vertices(self, points_S, canvas):
        # Polygon triangulation
        triangles = np.asarray(tripy.earclip(points_S))
        triangles = np.squeeze(triangles.reshape((1, -1, 2)))
        # Create texture coordinates
        triangles_gl = canvas.S2GL(triangles.T).T
        triangles_texture = triangles / canvas.size
        triangles_texture[:, 1] = 1.0 - triangles_texture[:, 1]  # Flip texture along y-axis
        return np.hstack((triangles_gl, triangles_texture)).astype('float32')


    def create_framebuffer(self, size, gl_internalformat, gl_format, gl_type):
        # Create a frame-buffer object
        framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
        # Create a texture for a frame-buffer
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, gl_internalformat, size[0], size[1], 0, gl_format, gl_type, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        # Return results
        return framebuffer, texture


    def set_cmap_buffer(self, cmap_name):
        cmap = GetColormapArray(cmap_name).astype('float32')
        # Create a buffer
        self.cmap_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.cmap_buffer)
        glBufferData(GL_UNIFORM_BUFFER, cmap.nbytes, cmap, GL_DYNAMIC_DRAW)
        # Set uniform block binding location
        cmap_buffer_block_index = glGetUniformBlockIndex(self.program_color, 'cmap')
        glUniformBlockBinding(self.program_color, cmap_buffer_block_index, 0)


    def update_cmap_buffer(self, cmap_name):
        cmap_array = GetColormapArray(cmap_name).astype('float32')
        # Update OpenGL framebuffers
        glBindBuffer(GL_UNIFORM_BUFFER, self.cmap_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, cmap_array.nbytes, cmap_array)


    def read_pixels(self, gl_framebuffer, framebuffer_size):
        # Initialize output image
        image_array = np.empty(shape=(framebuffer_size[0] * framebuffer_size[1] * 3), dtype='uint8')
        # Read pixels from the selected framebuffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, gl_framebuffer)
        glReadPixels(0, 0, framebuffer_size[0], framebuffer_size[1], GL_RGB, GL_UNSIGNED_BYTE, image_array)
        # Reshape and save the image
        return image_array.reshape((framebuffer_size[1], framebuffer_size[0], 3))


    def get_info_text(self):
        text = (f'CMAP = {self.cmap_names[self.cmap_id]}\n'
                f'ZOOM = {(self.canvas.scale_abs / self.canvas.scale_abs_default):.2E}\n'
                f'WINDOW = {self.window_size[0]}x{self.window_size[1]}\n'
                f'RENDER = {self.render_size[0]}x{self.render_size[1]}\n'
                f'SCALE = {self.pix_scale}\n'
                f'ITER = {self.num_iter}\n'
                f'FPS = {int(np.round((1.0 / self.clock.frame_time)))}\n')
        return text


    def draw_call(self):

        # 01. COMPUTE MANDELBROT ITERATIONS
        glViewport(0, 0, self.render_size[0], self.render_size[1])
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.framebuffer_mandelbrot)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program_mandelbrot)
        # Send uniforms to the GPU
        glUniform2dv(self.uniform_locations_mandelbrot['range_x'], 1, self.canvas.range_x.astype('float64'))
        glUniform2dv(self.uniform_locations_mandelbrot['range_y'], 1, self.canvas.range_y.astype('float64'))
        glUniform1d(self.uniform_locations_mandelbrot['pix_size'], 1.0 / self.canvas.scale_abs)
        glUniform1i(self.uniform_locations_mandelbrot['num_iter'], self.num_iter)
        # Draw geometry
        glBindVertexArray(self.polygon_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_polygon_vertices)

        # 02. FRACTAL COLORING
        glViewport(0, 0, self.render_size[0], self.render_size[1])
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.framebuffer_color)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program_color)
        # Bind resources
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.cmap_buffer)
        glBindTexture(GL_TEXTURE_2D, self.texture_mandelbrot)
        glActiveTexture(GL_TEXTURE0)
        # Send uniforms to the GPU
        glUniform1i(self.uniform_locations_color['num_iter'], self.num_iter)
        # Draw geometry
        glBindVertexArray(self.polygon_vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_polygon_vertices)

        # 03. COPY FRAMEBUFFERS
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.framebuffer_color)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, self.render_size[0], self.render_size[1],
                          0, 0, self.window_size[0], self.window_size[1],
                          GL_COLOR_BUFFER_BIT, GL_NEAREST)

        # 04. RENDER TEXT TO WINDOW
        if self.show_info_text:
            glViewport(0, 0, self.window_size[0], self.window_size[1])
            self.text_render.DrawText(self.get_info_text(), 10, 8, 1.0, (255, 255, 255))

        # 05. SWAP BUFFERS AND UPDATE TIMINGS
        glfw.swap_buffers(self.window)
        self.clock.Update()


'''
O------------------------------------------------------------------------------O
| AUXILIARY FUNCTIONS                                                          |
O------------------------------------------------------------------------------O
'''

def glfw_get_current_window_monitor(glfw_window):
    # Get all available monitors
    monitors = list(glfw.get_monitors())
    num_monitors = len(monitors)
    if num_monitors == 1:
        return monitors[0]
    # Get window bounding box
    window_TL = np.asarray(glfw.get_window_pos(glfw_window))
    window_BR = window_TL + np.asarray(glfw.get_window_size(glfw_window))
    # Loop over all monitors to find the one with largest overlapping area
    overlap_area = np.empty(num_monitors, dtype='int')
    for i in range(num_monitors):
        # Get monitor bounding box
        video_mode = glfw.get_video_mode(monitors[i])
        monitor_TL = np.asarray(glfw.get_monitor_pos(monitors[i]))
        monitor_BR = monitor_TL + np.asarray(video_mode.size)
        # Window-monitor overlap area
        min_x = np.maximum(window_TL[0], monitor_TL[0])
        max_x = np.minimum(window_BR[0], monitor_BR[0])
        min_y = np.maximum(window_TL[1], monitor_TL[1])
        max_y = np.minimum(window_BR[1], monitor_BR[1])
        overlap_area[i] = (max_x - min_x) * (max_y - min_y)
    # Return monitor with the highest overlap
    max_id = np.argmax(overlap_area)
    return monitors[max_id]
