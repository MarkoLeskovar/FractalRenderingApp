import os
import tripy
import numpy as np
from PIL import Image

# OpenGL modules
import glfw
import glfw.GLFW as GLFW_VAR
from OpenGL.GL import *

# Add custom modules
from .clock import ClockGLFW
from .default_config import *
from .render_text import TextRender
from .render_texture import RenderTexture
from .color import GetColormapArray
from .render_canvas import RenderCanvas
from .shader_utils import create_shader_program, read_shader_source, get_uniform_locations


# TODO : Add RenderCanvasManager class for automatic canvas handling


'''
O------------------------------------------------------------------------------O
| MAIN OPENGL APP CLASS                                                        |
O------------------------------------------------------------------------------O
'''

class FractalRenderingApp:

    # "Static" variables
    path_to_shaders = os.path.join(os.path.dirname(__file__), 'shaders')
    path_to_assets = os.path.join(os.path.dirname(__file__), 'assets')

    def __init__(self, app_config=None, fractal_config=None, controls_config=None, output_dir=None, cmaps=None):

        # Set app configuration variables
        self.app_cnfg = set_default_if_none(DEFAULT_APP_CONFIG, app_config)
        self.controls_cnfg = set_default_if_none(DEFAULT_CONTROLS_CONFIG, controls_config)
        self.fractal_cnfg = set_default_if_none(DEFAULT_FRACTAL_CONFIG, fractal_config)
        self.output_dir = set_default_if_none(DEFAULT_OUTPUT_DIR, output_dir)
        self.cmaps = set_default_if_none(DEFAULT_CMAPS, cmaps)
        self.cmap_id = 0

        # Fractal interation settings
        self.num_iter = int(self.fractal_cnfg['MANDELBROT']['NUM_ITER'])
        self.num_iter_min = int(self.fractal_cnfg['MANDELBROT']['NUM_ITER_MIN'])
        self.num_iter_max = int(self.fractal_cnfg['MANDELBROT']['NUM_ITER_MAX'])
        self.num_iter_step = int(self.fractal_cnfg['MANDELBROT']['NUM_ITER_STEP'])

        # Pixel scaling settings
        self.pix_scale_min = float(self.app_cnfg['PIX_SCALE_MIN'])
        self.pix_scale_max = float(self.app_cnfg['PIX_SCALE_MAX'])
        self.pix_scale_step = float(self.app_cnfg['PIX_SCALE_STEP'])

        # Initialize GLFW
        glfw.init()
        # Create GLFW window and set the icon
        self.window_vsync = True
        window_size = (int(self.app_cnfg['WIN_WIDTH']), int(self.app_cnfg['WIN_HEIGHT']))
        self.window = self._create_main_window(window_size, self.window_vsync)
        icon = Image.open(os.path.join(self.path_to_assets, 'mandelbrot.png')).resize((256, 256))
        glfw.set_window_icon(self.window, 1, icon)

        # Get the actual window size
        self.pix_scale = float(glfw.get_window_content_scale(self.window)[0])
        self.window_size = np.asarray(glfw.get_framebuffer_size(self.window)).astype('int')
        self.render_size = (self.window_size / self.pix_scale).astype('int')

        # Create a render canvas
        range_x = (float(self.fractal_cnfg['MANDELBROT']['RANGE_X_MIN']),
                   float(self.fractal_cnfg['MANDELBROT']['RANGE_X_MAX']))

        # TODO : DEBUG
        self.canvas_pos = np.asarray([100, 100])
        self.canvas_size = (np.asarray([0.5 * self.window_size[0], self.window_size[1]]) - 2 * self.canvas_pos).astype('int')
        self.canvas = RenderCanvas(self.canvas_pos, self.canvas_size, self.pix_scale, range_x)

        # Create framebuffers
        self.canvas.add_framebuffer('ITER', GL_R32F, GL_RED, GL_FLOAT)
        self.canvas.add_framebuffer('COLOR', GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)

        # Blit texture class
        self.blit_texture_to_screen = RenderTexture()

        # Create GLFW clock
        self.clock = ClockGLFW()

        # Create a text renderer
        self.text_size = int(self.app_cnfg['FONT_SIZE'])
        self.text_file = os.path.join(self.path_to_assets, self.app_cnfg['FONT_FILE'])
        self.text_render = TextRender()
        self.text_render.set_font(self.text_file, self.text_size * self.pix_scale)
        self.text_render.set_window_size(self.window_size)

        # Set GLFW callback functions
        self._set_callback_functions_glfw()

        # Read shader source code
        base_vert_source = read_shader_source(os.path.join(self.path_to_shaders, 'fractal_base.vert'))
        color_frag_source = read_shader_source(os.path.join(self.path_to_shaders, 'fractal_color.frag'))
        mandelbrot_frag_source = read_shader_source(os.path.join(self.path_to_shaders, 'fractal_mandelbrot.frag'))
        # Create shader programs
        self.program_color = create_shader_program(base_vert_source, color_frag_source)
        self.program_mandelbrot = create_shader_program(base_vert_source, mandelbrot_frag_source)

        # Get uniform locations
        self.uniform_locations_color = get_uniform_locations(self.program_color, ['num_iter'])
        self.uniform_locations_mandelbrot = get_uniform_locations(
            self.program_mandelbrot, ['pix_size', 'mouse_pos', 'range_x', 'range_y', 'num_iter'])

        # Create buffers
        self._set_cmap_buffer(self.cmaps[self.cmap_id])


    def run(self):
        # Main app loop
        self.window_open = True
        while self.window_open:
            # Draw call
            if not self.window_minimized:
                self._render_call()
            # Event handling
            glfw.poll_events()
            self._process_hold_keys()
            self.canvas.update_mouse_pos()


    def close(self):
        # Delete custom classes
        self.canvas.delete()
        self.text_render.delete()
        self.blit_texture_to_screen.delete()
        # Delete OpenGL buffers
        glDeleteBuffers(1, [self.cmap_buffer])
        glDeleteProgram(self.program_mandelbrot)
        glDeleteProgram(self.program_color)
        # Terminate GLFW
        glfw.destroy_window(self.window)
        glfw.terminate()


    @classmethod
    def set_path_to_assets(cls, path: str):
        cls.path_to_assets = str(path)


    # O------------------------------------------------------------------------------O
    # | GLFW EVENT HANDLING - CALLBACK FUNCTIONS AND USER INPUT                      |
    # O------------------------------------------------------------------------------O

    def _set_callback_functions_glfw(self):
        # Toggle flags
        self.window_open = True
        self.window_minimized = False
        self.window_fullscreen = False
        self.keyboard_up_key_hold = False
        self.keyboard_down_key_hold = False
        self.mouse_left_button_hold = False
        self.show_info_text = True
        # Window callback functions
        glfw.set_window_close_callback(self.window, self._callback_window_close)
        glfw.set_window_size_callback(self.window, self._callback_window_resize)
        glfw.set_window_iconify_callback(self.window, self._callback_window_minimized)
        glfw.set_window_content_scale_callback(self.window, self._callback_content_scale)
        # User input callback functions
        glfw.set_cursor_pos_callback(self.window, self._callback_cursor_position)
        glfw.set_mouse_button_callback(self.window, self._callback_mouse_button)
        glfw.set_scroll_callback(self.window, self._callback_mouse_scroll)
        glfw.set_key_callback(self.window, self._callback_keyboad_button)


    def _callback_window_close(self, window):
        self.window_open = False


    def _callback_window_minimized(self, window, iconified):
        self.window_minimized = bool(iconified)


    def _callback_window_resize(self, window, width, height):
        if not self.window_minimized:
            temp_size = glfw.get_framebuffer_size(self.window)
            self._update_window_size(temp_size, self.pix_scale)
            self.text_render.set_window_size((width, height))
            self._render_call()


    def _callback_content_scale(self, window, scale_x, scale_y):
        self._update_window_size(self.window_size, scale_x)
        self.text_render.set_font(self.text_file, self.text_size * self.pix_scale)
        self._render_call()


    def _callback_keyboad_button(self, window, key, scancode, action, mods):
        # Exit the app
        if key == getattr(glfw, self.controls_cnfg['EXIT']) and action == glfw.PRESS:
            self.window_open = False

        # Show into text
        if key == getattr(glfw, self.controls_cnfg['INFO']) and action == glfw.PRESS:
            self.show_info_text = not self.show_info_text

        # Toggle fullscreen
        if key == getattr(glfw, self.controls_cnfg['FULLSCREEN']) and action == glfw.PRESS:
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
        if key == getattr(glfw, self.controls_cnfg['ITER_INCREASE']) and action == glfw.PRESS:
            self.num_iter = min(self.num_iter + self.num_iter_step, self.num_iter_max)


        # Decrease number of iterations
        if key == getattr(glfw, self.controls_cnfg['ITER_DECREASE']) and action == glfw.PRESS:
            self.num_iter = max(self.num_iter - self.num_iter_step, self.num_iter_min)


        # Reset shift and scale
        if key == getattr(glfw, self.controls_cnfg['RESET_VIEW']) and action == glfw.PRESS:
            self.canvas.reset_shift_and_scale()
            # Also reset the number of fractals iterations
            self.num_iter = self.num_iter_min

        # Increase pixel scale
        if key == getattr(glfw, self.controls_cnfg['PIX_SCALE_INCREASE']) and action == glfw.PRESS:
            temp_pix_scale = min(self.pix_scale + self.pix_scale_step, self.pix_scale_max)
            self._update_window_size(self.window_size, temp_pix_scale)

        # Decrease pixel scale
        if key == getattr(glfw, self.controls_cnfg['PIX_SCALE_DECREASE']) and action == glfw.PRESS:
            temp_pix_scale = max(self.pix_scale - self.pix_scale_step, self.pix_scale_min)
            self._update_window_size(self.window_size, temp_pix_scale)

        # Hold zoom-in
        if key == getattr(glfw, self.controls_cnfg['ZOOM_IN']):
            if action == glfw.PRESS:
                self.keyboard_up_key_hold = True
            elif action == glfw.RELEASE:
                self.keyboard_up_key_hold = False

        # Hold zoom-out
        if key == getattr(glfw, self.controls_cnfg['ZOOM_OUT']):
            if action == glfw.PRESS:
                self.keyboard_down_key_hold = True
            elif action == glfw.RELEASE:
                self.keyboard_down_key_hold = False

        # Next colormap
        if key == getattr(glfw, self.controls_cnfg['CMAP_NEXT']) and action == glfw.PRESS:
            self.cmap_id += 1
            if self.cmap_id >= len(self.cmaps):
                self.cmap_id = 0
            self._update_cmap_buffer(self.cmaps[self.cmap_id])

        # Previous colormap
        if key == getattr(glfw, self.controls_cnfg['CMAP_PREV']) and action == glfw.PRESS:
            self.cmap_id -= 1
            if self.cmap_id < 0:
                self.cmap_id = len(self.cmaps) - 1
            self._update_cmap_buffer(self.cmaps[self.cmap_id])

        # Toggle vsync for uncapped frame rate
        if key == getattr(glfw, self.controls_cnfg['VSYNC']) and action == glfw.PRESS:
            self.window_vsync = not self.window_vsync
            glfw.swap_interval(int(self.window_vsync))

        # Save a screenshot
        if key == getattr(glfw, self.controls_cnfg['SCREENSHOT']) and action == glfw.PRESS:
            os.makedirs(self.output_dir, exist_ok=True)
            # Get current screenshot counter
            counter = 1
            output_files = os.listdir(self.output_dir)
            if len(output_files) > 0:
                temp_list = sorted([int(x.split('.')[0]) for x in output_files])
                counter = temp_list[-1] + 1
            # Read pixels and save the image
            output_image = self.read_pixels()
            image_pil = Image.fromarray(np.flipud(output_image))
            image_pil.save(os.path.join(self.output_dir, f'{counter}.png'))

    def read_pixels(self):
        image_array = np.empty(shape=(self.window_size[0] * self.window_size[1] * 3), dtype='uint8')
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glReadPixels(0, 0, self.window_size[0], self.window_size[1], GL_RGB, GL_UNSIGNED_BYTE, image_array)
        return image_array.reshape((self.window_size[1], self.window_size[0], 3))

    def _callback_mouse_button(self, window, button, action, mod):
        # Hold down the mouse button
        if button == getattr(glfw, self.controls_cnfg['SHIFT_VIEW']):
            if action == glfw.PRESS:
                self.mouse_left_button_hold = True
            elif action == glfw.RELEASE:
                self.mouse_left_button_hold = False


    def _callback_mouse_scroll(self, window, x_offset, y_offset):
        # Zoom-in
        if y_offset > 0:
            temp_scale_step = 5.0 * self.canvas.scale_step * abs(y_offset)
            self.canvas.increase_scale(temp_scale_step)
        # Zoom-out
        if y_offset < 0:
            temp_scale_step = 5.0 * self.canvas.scale_step * abs(y_offset)
            self.canvas.decrease_scale(temp_scale_step)


    def _callback_cursor_position(self, window, x_pos, y_pos):
        self.canvas.mouse_pos = (x_pos, y_pos)


    def _process_hold_keys(self):
        # Pan screen
        if self.mouse_left_button_hold:
            self.canvas.update_shift()
        # Zoom-in
        if self.keyboard_up_key_hold:
            temp_scale_step = self.canvas.scale_step * self.clock.frame_time * 60.0
            self.canvas.increase_scale(temp_scale_step)
        # Zoom-out
        if self.keyboard_down_key_hold:
            temp_scale_step = self.canvas.scale_step * self.clock.frame_time * 60.0
            self.canvas.decrease_scale(temp_scale_step)


    def _update_window_size(self, size, pix_scale):
        # Update window size
        self.window_size = np.asarray(size).astype('int')
        self.pix_scale = float(pix_scale)
        self.render_size = (self.window_size / self.pix_scale).astype('int')
        # Update app variables
        self.canvas_size = (np.asarray([0.5 * self.window_size[0], self.window_size[1]]) - 2 * self.canvas_pos).astype('int')
        self.canvas.resize(self.canvas_size, self.pix_scale)


    # O------------------------------------------------------------------------------O
    # | OPENGL FUNCTIONS                                                             |
    # O------------------------------------------------------------------------------O

    def _create_main_window(self, size, vsync=True):
        size = np.asarray(size).astype('int')
        # GLFW window settings
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


    def _set_cmap_buffer(self, cmap_name):
        cmap = GetColormapArray(cmap_name).astype('float32')
        # Create a buffer
        self.cmap_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.cmap_buffer)
        glBufferData(GL_UNIFORM_BUFFER, cmap.nbytes, cmap, GL_DYNAMIC_DRAW)
        # Set uniform block binding location
        cmap_buffer_block_index = glGetUniformBlockIndex(self.program_color, 'cmap')
        glUniformBlockBinding(self.program_color, cmap_buffer_block_index, 0)


    def _update_cmap_buffer(self, cmap_name):
        cmap_array = GetColormapArray(cmap_name).astype('float32')
        # Update OpenGL framebuffers
        glBindBuffer(GL_UNIFORM_BUFFER, self.cmap_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, cmap_array.nbytes, cmap_array)


    def _get_info_text(self):
        text = (
            f'CMAP = {self.cmaps[self.cmap_id]}\n'
            f'ZOOM = {self.canvas.scale_rel:.2E}\n'
            f'WINDOW = {self.window_size[0]}x{self.window_size[1]}\n'
            f'CANVAS = {self.canvas.size[0]}x{self.canvas.size[1]}\n'
            f'RENDER = {self.canvas.render_size[0]}x{self.canvas.render_size[1]}\n'
            f'SCALE = {self.pix_scale}\n'
            f'ITER = {self.num_iter}\n'
            f'FPS = {int(np.round(self.clock.frame_rate))}\n'
            f'ACTIVE = {self.canvas.is_active()}'
        )
        return text


    def _render_fractal_program(self, canvas, gl_program, uniform_locations, num_iter):

        # 00. VIEWPORT SIZE
        glViewport(0, 0, canvas.render_size[0], canvas.render_size[1])

        # 01. FRACTAL ITERATIONS
        glBindFramebuffer(GL_FRAMEBUFFER, canvas.framebuffer['ITER'].id)
        glUseProgram(gl_program)
        # Send uniforms to the GPU
        mouse_pos_w = canvas.s2w(canvas.mouse_pos)
        glUniform2dv(uniform_locations['mouse_pos'], 1, mouse_pos_w.astype('float64'))
        glUniform2dv(uniform_locations['range_x'], 1, canvas.range_x.astype('float64'))
        glUniform2dv(uniform_locations['range_y'], 1, canvas.range_y.astype('float64'))
        glUniform1d(uniform_locations['pix_size'], 1.0 / canvas.scale_abs)
        glUniform1i(uniform_locations['num_iter'], num_iter)
        # Draw arrays
        glBindVertexArray(canvas._polygon_vao)
        glDrawArrays(GL_TRIANGLES, 0, canvas._polygon_buffer_n_indices)

        # 02. FRACTAL COLOR
        glBindFramebuffer(GL_FRAMEBUFFER, canvas.framebuffer['COLOR'].id)
        glUseProgram(self.program_color)
        # Bind resources
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.cmap_buffer)
        glBindTexture(GL_TEXTURE_2D, canvas.framebuffer['ITER'].tex_id)
        glActiveTexture(GL_TEXTURE0)
        # Send uniforms to the GPU
        glUniform1i(self.uniform_locations_color['num_iter'], num_iter)
        # Draw geometry
        glBindVertexArray(canvas._polygon_vao)
        glDrawArrays(GL_TRIANGLES, 0, canvas._polygon_buffer_n_indices)


    def _render_call(self):

        # Clear the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClearColor(0.0, 0.5, 0.5, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Render the Mandelbrot set
        self._render_fractal_program(self.canvas, self.program_mandelbrot, self.uniform_locations_mandelbrot, self.num_iter)

        # Blit to screen
        self.blit_texture_to_screen(self.window_size, self.canvas.pos, self.canvas.size, self.canvas.framebuffer['COLOR'].tex_id)

        # Render text to screen
        if self.show_info_text:
            self.text_render.draw_text(self._get_info_text(), (10, 8), 1.0, (255, 255, 255))

        # Swap buffers and update timings
        glfw.swap_buffers(self.window)
        self.clock.update()


'''
O------------------------------------------------------------------------------O
| AUXILIARY FUNCTIONS                                                          |
O------------------------------------------------------------------------------O
'''

def set_default_if_none(default_value, value=None):
    output_value = default_value
    if value is not None:
        output_value = value
    return output_value


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
