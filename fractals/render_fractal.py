import os
import numpy as np
from PIL import Image

# OpenGL modules
import glfw
import glfw.GLFW as GLFW_VAR
from OpenGL.GL import *

# Add custom modules
from .clock import Clock
from .config import DEFAULT
from .render_text import RenderText
from .render_texture import RenderTexture
from .color import get_colormap_array
from .render_canvas import RenderCanvas
from .shader_utils import create_shader_program, read_shader_source, get_uniform_locations

'''
O------------------------------------------------------------------------------O
| MAIN OPENGL APP CLASS                                                        |
O------------------------------------------------------------------------------O
'''

# TODO : Implement support for user-defined max framerate !!
# TODO : Keep zooming speed consistent when switching between high and low framerate !!
# TODO : Add support to display the number of iterations under the mouse cursor !!

class FractalRenderingApp:

    # "Static" variables
    path_to_shaders = os.path.join(os.path.dirname(__file__), 'shaders')
    path_to_assets = os.path.join(os.path.dirname(__file__), 'assets')

    def __init__(self, window_config=None, fractal_config=None, control_config=None, output_dir=None, cmaps=None):

        # Set configuration variables
        self.window_config = default_if_none(DEFAULT.window_config, window_config)
        self.fractal_config = default_if_none(DEFAULT.fractal_config, fractal_config)
        self.controls_config = default_if_none(DEFAULT.controls_config, control_config)
        self.output_dir = default_if_none(DEFAULT.output_dir, output_dir)
        self.cmaps = default_if_none(DEFAULT.cmaps, cmaps)
        self.cmap_id = 0

        # Assign correct control values
        self.control_ids = {}
        for key, value in self.controls_config.items():
            if hasattr(glfw, value):
                self.control_ids[key] = 'key'
            elif isinstance(value, str):
                self.control_ids[key] = 'char'
                self.controls_config[key] = value.lower()
            else:
                raise ValueError('Invalid control type')

        # Create GLFW window and set the icon
        glfw.init()
        self.window_vsync = True
        window_size = (int(self.window_config['WIDTH']), int(self.window_config['HEIGHT']))
        self.window = create_main_window(window_size, self.window_vsync)
        icon = Image.open(os.path.join(self.path_to_assets, 'mandelbrot.png')).resize((256, 256))
        glfw.set_window_icon(self.window, 1, icon)

        # Get the actual window size
        self.pix_scale = float(glfw.get_window_content_scale(self.window)[0])
        self.window_size = np.asarray(glfw.get_framebuffer_size(self.window)).astype('int')

        # Create render canvases
        self.canvas_names = ['MANDELBROT', 'JULIA']
        self.canvas_list = []
        for canvas_name in self.canvas_names:
            temp_x_min = self.fractal_config[canvas_name]['RANGE_X_MIN']
            temp_x_max = self.fractal_config[canvas_name]['RANGE_X_MAX']
            temp_canvas = RenderCanvas((0, 0), self.window_size, self.pix_scale, (temp_x_min, temp_x_max))
            temp_canvas.init()
            temp_canvas.add_framebuffer('ITER', GL_R32F, GL_RED, GL_FLOAT)
            temp_canvas.add_framebuffer('COLOR', GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)
            self.canvas_list.append(temp_canvas)

        # Active canvas
        self.render_mode_id = 0
        self.canvas_id = 0

        # Texture render class
        self.render_texture = RenderTexture()
        self.render_texture.init()

        # Create GLFW clock
        self.clock = Clock()

        # Text render class
        self.text_id = 2
        self.text_file = os.path.join(self.path_to_assets, self.window_config['FONT_FILE'])
        self.render_text = RenderText()
        self.render_text.init()
        self.render_text.set_window_size(self.window_size)
        self.render_text.set_font(self.text_file, self.window_config['FONT_SIZE'] * self.pix_scale)

        # Set GLFW callback functions
        self._set_action_handlers()
        self._set_callback_functions()

        # Read shader source code
        base_vert_source = read_shader_source(os.path.join(self.path_to_shaders, 'fractal_base.vert'))
        color_frag_source = read_shader_source(os.path.join(self.path_to_shaders, 'fractal_color.frag'))
        julia_frag_source = read_shader_source(os.path.join(self.path_to_shaders, 'fractal_julia.frag'))
        mandel_frag_source = read_shader_source(os.path.join(self.path_to_shaders, 'fractal_mandelbrot.frag'))

        # Create shader programs
        self.program_color = create_shader_program(base_vert_source, color_frag_source)
        self.program_julia = create_shader_program(base_vert_source, julia_frag_source)
        self.program_mandel = create_shader_program(base_vert_source, mandel_frag_source)

        # Get uniform locations
        self.uniforms_color = get_uniform_locations(self.program_color, ['num_iter'])
        self.uniforms_julia = get_uniform_locations(self.program_julia, ['pix_size', 'mouse_pos', 'range_x', 'range_y', 'num_iter'])
        self.uniforms_mandel = get_uniform_locations(self.program_mandel, ['pix_size', 'mouse_pos', 'range_x', 'range_y', 'num_iter'])

        # Create buffers
        self._set_cmap_buffer(self.cmaps[self.cmap_id])

        # Run the app to determine initial framerate
        # self.prev_frame_time = 0.0
        # self.init_run(num_runs=10)


    # def init_run(self, num_runs=10):
    #     update_interval = self.clock.update_interval
    #     self.clock.update_interval = 0.02
    #     for i in range(num_runs):
    #         self._render_call()
    #         for canvas in self.canvas_list:
    #             canvas.needs_update = self.clock.frame_time
    #     self.prev_frame_time = self.clock.frame_time
    #     self.clock.update_interval = update_interval


    def run(self):
        while self.window_open:
            # Draw call
            if not self.window_minimized:
                self._render_call()
            # Event handling
            glfw.poll_events()
            self._process_hold_keys()
            # Update mouse pos
            for canvas in self.canvas_list:
                canvas.update_mouse_pos()


    def close(self):
        # Delete custom classes
        self.render_text.delete()
        self.render_texture.delete()
        for canvas in self.canvas_list:
            canvas.delete()
        # Delete shader programs
        glDeleteProgram(self.program_color)
        glDeleteProgram(self.program_julia)
        glDeleteProgram(self.program_mandel)
        # Delete OpenGL buffers
        glDeleteBuffers(1, [self.cmap_buffer])
        # Terminate GLFW
        glfw.destroy_window(self.window)
        glfw.terminate()


    # O------------------------------------------------------------------------------O
    # | GLFW EVENT HANDLING - USER ACTIONS                                           |
    # O------------------------------------------------------------------------------O

    def _set_action_handlers(self):
        self.action_handlers = {
            "EXIT":                 self._action_exit,
            "INFO":                 self._action_info_text,
            "VSYNC":                self._action_toggle_vsync,
            "FULLSCREEN":           self._action_toggle_fullscreen,
            "SCREENSHOT":           self._action_take_screenshot,
            "ZOOM_IN":              self._action_hold_zoom_in,
            "ZOOM_OUT":             self._action_hold_zoom_out,
            "RESET_VIEW":           self._action_reset_view,
            "ITER_INCREASE":        self._action_increase_num_iter,
            "ITER_DECREASE":        self._action_decrease_num_iter,
            "SCALE_INCREASE":       self._action_increase_pixel_scale,
            "SCALE_DECREASE":       self._action_decrease_pixel_scale,
            "CMAP_NEXT":            self._action_next_colormap,
            "CMAP_PREV":            self._action_prev_colormap,
            "JULIA_SET":            self._action_switch_fractals
        }


    def _action_exit(self, action_type):
        if action_type == glfw.PRESS:
            self.window_open = False


    def _action_info_text(self, action_type):
        if action_type == glfw.PRESS:
            self.text_id = (self.text_id + 1) % 4


    def _action_toggle_fullscreen(self, action_type):
        if action_type == glfw.PRESS:
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


    def _action_increase_num_iter(self, action_type):
        if action_type == glfw.PRESS or action_type == glfw.REPEAT:
            temp_dict = self.fractal_config[self.canvas_names[self.canvas_id]]
            temp_num_iter = min(temp_dict['N_ITER'] + temp_dict['N_ITER_STEP'], temp_dict['N_ITER_MAX'])
            self.fractal_config[self.canvas_names[self.canvas_id]]['N_ITER'] = int(temp_num_iter)
            self.canvas_list[self.canvas_id].needs_update = True

    def _action_decrease_num_iter(self, action_type):
        if action_type == glfw.PRESS or action_type == glfw.REPEAT:
            canvas_name = self.canvas_names[self.canvas_id]
            temp_dict = self.fractal_config[canvas_name]
            temp_num_iter = max(temp_dict['N_ITER'] - temp_dict['N_ITER_STEP'], temp_dict['N_ITER_MIN'])
            self.fractal_config[canvas_name]['N_ITER'] = int(temp_num_iter)
            self.canvas_list[self.canvas_id].needs_update = True


    def _action_reset_view(self, action_type):
        if action_type == glfw.PRESS:
            canvas_name = self.canvas_names[self.canvas_id]
            self.canvas_list[self.canvas_id].reset_shift_and_scale()
            # Reset number of iterations
            temp_num_iter = self.fractal_config[canvas_name]['N_ITER_MIN']
            self.fractal_config[canvas_name]['N_ITER'] = int(temp_num_iter)


    def _action_increase_pixel_scale(self, action_type):
        if action_type == glfw.PRESS or action_type == glfw.REPEAT:
            temp_pix_scale = self.pix_scale + self.window_config['SCALE_STEP']
            temp_pix_scale = min(temp_pix_scale, self.window_config['SCALE_MAX'])
            self._update_window_size(self.window_size, temp_pix_scale)


    def _action_decrease_pixel_scale(self, action_type):
        if action_type == glfw.PRESS or action_type == glfw.REPEAT:
            temp_pix_scale = self.pix_scale - self.window_config['SCALE_STEP']
            temp_pix_scale = max(temp_pix_scale, self.window_config['SCALE_MIN'])
            self._update_window_size(self.window_size, temp_pix_scale)


    def _action_hold_zoom_in(self, action_type):    # Hold zoom-in
        if action_type == glfw.PRESS:
            self.hold_zoom_in = True
        elif action_type == glfw.RELEASE:
            self.hold_zoom_in = False


    def _action_hold_zoom_out(self, action_type):
        if action_type == glfw.PRESS:
            self.hold_zoom_out = True
        elif action_type == glfw.RELEASE:
            self.hold_zoom_out = False


    def _action_next_colormap(self, action_type):
        if action_type == glfw.PRESS or action_type == glfw.REPEAT:
            self.cmap_id = (self.cmap_id + 1) % len(self.cmaps)
            self._update_cmap_buffer(self.cmaps[self.cmap_id])
            for canvas in self.canvas_list:
                canvas.needs_update = True

    def _action_prev_colormap(self, action_type):
        if action_type == glfw.PRESS or action_type == glfw.REPEAT:
            self.cmap_id = (self.cmap_id - 1) % len(self.cmaps)
            self._update_cmap_buffer(self.cmaps[self.cmap_id])
            for canvas in self.canvas_list:
                canvas.needs_update = True


    def _action_toggle_vsync(self, action_type):
        if action_type == glfw.PRESS:
            self.window_vsync = not self.window_vsync
            glfw.swap_interval(int(self.window_vsync))


    def _action_take_screenshot(self, action_type):
        if action_type == glfw.PRESS:
            os.makedirs(self.output_dir, exist_ok=True)
            # Get current screenshot counter
            counter = 1
            output_files = os.listdir(self.output_dir)
            if len(output_files) > 0:
                temp_list = sorted([int(out_file.split('.')[0]) for out_file in output_files])
                counter = temp_list[-1] + 1
            # Read pixels and save the image
            output_image = self._read_pixels()
            image_pil = Image.fromarray(np.flipud(output_image))
            image_pil.save(os.path.join(self.output_dir, f'{counter}.png'))


    def _action_switch_fractals(self, action_type):
        if action_type == glfw.PRESS:
            self.render_mode_id = (self.render_mode_id + 1) % 3
            self._set_canvas_sizes()
            self._set_canvas_settings()


    # O------------------------------------------------------------------------------O
    # | GLFW EVENT HANDLING - CALLBACK FUNCTIONS                                     |
    # O------------------------------------------------------------------------------O

    def _set_callback_functions(self):
        # Toggle flags
        self.window_open = True
        self.window_minimized = False
        self.window_fullscreen = False
        self.hold_zoom_in = False
        self.hold_zoom_out = False
        self.mouse_button_hold = False
        # Window callback functions
        glfw.set_window_close_callback(self.window, self._callback_window_close)
        glfw.set_window_size_callback(self.window, self._callback_window_resize)
        glfw.set_window_iconify_callback(self.window, self._callback_window_minimized)
        glfw.set_window_content_scale_callback(self.window, self._callback_window_scale)
        # User input callback functions
        glfw.set_cursor_pos_callback(self.window, self._callback_mouse_position)
        glfw.set_mouse_button_callback(self.window, self._callback_mouse_button)
        glfw.set_scroll_callback(self.window, self._callback_mouse_scroll)
        glfw.set_char_callback(self.window, self._callback_char_button)
        glfw.set_key_callback(self.window, self._callback_key_button)


    def _callback_window_close(self, window):
        self.window_open = False


    def _callback_window_minimized(self, window, iconified):
        self.window_minimized = bool(iconified)


    def _callback_window_resize(self, window, width, height):
        if not self.window_minimized:
            temp_size = glfw.get_framebuffer_size(self.window)
            self._update_window_size(temp_size, self.pix_scale)
            self.render_text.set_window_size((width, height))
            self._render_call()


    def _callback_window_scale(self, window, scale_x, scale_y):
        self._update_window_size(self.window_size, scale_x)
        self.render_text.set_font(self.text_file, self.window_config['FONT_SIZE'] * self.pix_scale)
        self._render_call()


    def _callback_key_button(self, window, key, scancode, action, mods):
        for action_name, binding in self.controls_config.items():
            if self.control_ids[action_name] == 'key' and key == getattr(glfw, binding):
                self.action_handlers[action_name](action)


    def _callback_char_button(self, windows, codepoint):
        ch = chr(codepoint)
        for action_name, binding in self.controls_config.items():
            if self.control_ids[action_name] == 'char' and ch == binding:
                self.action_handlers[action_name](glfw.PRESS)


    def _callback_mouse_button(self, window, button, action, mod):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.mouse_button_hold = True
            elif action == glfw.RELEASE:
                self.mouse_button_hold = False


    def _callback_mouse_scroll(self, window, x_offset, y_offset):
        canvas = self.canvas_list[self.canvas_id]
        scale_step = 5.0 * canvas.scale_step * abs(y_offset)
        # Zoom-in
        if y_offset > 0:
            canvas.increase_scale(scale_step)
        # Zoom-out
        if y_offset < 0:
            canvas.decrease_scale(scale_step)


    def _callback_mouse_position(self, window, x_pos, y_pos):
        for canvas in self.canvas_list:
            canvas.mouse_pos = (x_pos, y_pos)
        # Update Julia set flag
        if self.render_mode_id == 1 and not self.mouse_button_hold:
            self.canvas_list[1].needs_update = True


    def _set_canvas_sizes(self):
        # Mandelbrot set
        if self.render_mode_id == 0:
            self.canvas_list[0].resize(self.window_size, self.pix_scale)
        # Mandelbrot and Julia set
        elif self.render_mode_id == 1:
            self.canvas_list[0].resize(self.window_size, self.pix_scale)
            self.canvas_list[1].resize(0.3 * self.window_size, self.pix_scale)
            self.canvas_list[1].pos = np.ceil(0.7 * self.window_size)
        # Julia set
        elif self.render_mode_id == 2:
            self.canvas_list[1].resize(self.window_size, self.pix_scale)
            self.canvas_list[1].pos = (0, 0)


    def _set_canvas_settings(self):
        # Mandelbrot set
        if self.render_mode_id == 0:
            self.fractal_config['JULIA']['N_ITER'] = self.fractal_config['JULIA']['N_ITER_MIN']
            self.fractal_config['MANDELBROT']['N_ITER'] = self.fractal_config['MANDELBROT']['N_ITER_MIN']
            self.canvas_list[1].reset_shift_and_scale()
            self.canvas_id = 0
        # Julia set
        elif self.render_mode_id == 2:
            self.canvas_list[0].reset_shift_and_scale()
            self.canvas_id = 1


    # TODO: Clock frame time needs to be replaced with a fixed value when using custom framerate
    def _process_hold_keys(self):
        canvas = self.canvas_list[self.canvas_id]
        # Pan screen
        if self.mouse_button_hold:
            canvas.update_shift()
        # Zoom-in
        if self.hold_zoom_in:
            scale_step = canvas.scale_step * self.clock.frame_time * 60.0
            canvas.increase_scale(scale_step)
        # Zoom-out
        if self.hold_zoom_out:
            scale_step = canvas.scale_step * self.clock.frame_time * 60.0
            canvas.decrease_scale(scale_step)


    def _update_window_size(self, size, pix_scale):
        self.window_size = np.asarray(size).astype('int')
        self.pix_scale = float(pix_scale)
        self._set_canvas_sizes()


    def _read_pixels(self):
        image_array = np.empty(shape=(self.window_size[0] * self.window_size[1] * 3), dtype='uint8')
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        glReadPixels(0, 0, self.window_size[0], self.window_size[1], GL_RGB, GL_UNSIGNED_BYTE, image_array)
        return image_array.reshape((self.window_size[1], self.window_size[0], 3))


    # O------------------------------------------------------------------------------O
    # | OPENGL FUNCTIONS                                                             |
    # O------------------------------------------------------------------------------O

    def _set_cmap_buffer(self, cmap_name):
        cmap = get_colormap_array(cmap_name).astype('float32')
        # Create a buffer
        self.cmap_buffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.cmap_buffer)
        glBufferData(GL_UNIFORM_BUFFER, cmap.nbytes, cmap, GL_DYNAMIC_DRAW)
        # Set uniform block binding location
        cmap_buffer_block_index = glGetUniformBlockIndex(self.program_color, 'cmap')
        glUniformBlockBinding(self.program_color, cmap_buffer_block_index, 0)


    def _update_cmap_buffer(self, cmap_name):
        cmap_array = get_colormap_array(cmap_name).astype('float32')
        # Update OpenGL framebuffers
        glBindBuffer(GL_UNIFORM_BUFFER, self.cmap_buffer)
        glBufferSubData(GL_UNIFORM_BUFFER, 0, cmap_array.nbytes, cmap_array)


    def _get_fractal_info_text(self):
        canvas = self.canvas_list[self.canvas_id]
        canvas_name = self.canvas_names[self.canvas_id]
        text = (
            f'FRACTAL = {canvas_name.lower()} set\n'
            f'COLORS = {self.cmaps[self.cmap_id]}\n'
            f'WINDOW = {self.window_size[0]}x{self.window_size[1]}\n'
            f'RENDER = {canvas.render_size[0]}x{canvas.render_size[1]}\n'
            f'SCALE = {self.pix_scale}\n'
            f'ZOOM = {canvas.scale_rel:.2E}\n'
            f'ITER = {self.fractal_config[canvas_name]["N_ITER"]}\n'
            f'FPS = {int(np.round(self.clock.frame_rate))}\n'
        )
        return text


    def _get_mouse_info_text(self):
        canvas = self.canvas_list[self.canvas_id]
        mouse_pos_w = canvas.s2w(canvas.mouse_pos)
        text = (
            f'MOUSE POS\n'
            f'Re = {mouse_pos_w[0]: .15f}\n'
            f'Im = {mouse_pos_w[1]: .15f}\n'
        )
        return text


    def _render_fractal_program(self, canvas, gl_program, uniforms, mouse_pos_w, num_iter):
        mouse_pos_w = np.asarray(mouse_pos_w)
        
        # 00. VIEWPORT SIZE
        glViewport(0, 0, canvas.render_size[0], canvas.render_size[1])

        # 01. FRACTAL ITERATIONS
        glBindFramebuffer(GL_FRAMEBUFFER, canvas.framebuffer['ITER'].id)
        glUseProgram(gl_program)
        # Send uniforms to the GPU
        glUniform2dv(uniforms['mouse_pos'], 1, mouse_pos_w.astype('float64'))
        glUniform2dv(uniforms['range_x'], 1, canvas.range_x.astype('float64'))
        glUniform2dv(uniforms['range_y'], 1, canvas.range_y.astype('float64'))
        glUniform1d(uniforms['pix_size'], 1.0 / canvas.scale_abs)
        glUniform1i(uniforms['num_iter'], num_iter)
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
        glUniform1i(self.uniforms_color['num_iter'], num_iter)
        # Draw geometry
        glBindVertexArray(canvas._polygon_vao)
        glDrawArrays(GL_TRIANGLES, 0, canvas._polygon_buffer_n_indices)


    def _render_call(self):

        # DEBUG INFO
        # print(f'Update MANDELBROT: {self.canvas_list[0].needs_update}, Update JULIA: {self.canvas_list[1].needs_update}')

        # if any(canvas.needs_update for canvas in self.canvas_list):
        #     self.prev_frame_time = self.clock.frame_time
        #     print(f'Frame time: {self.prev_frame_time }')

        # Clear the screen
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClearColor(0.0, 0.5, 0.5, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Update fractal constant
        if self.render_mode_id == 1:
            self.fractal_constant = self.canvas_list[0].s2w(self.canvas_list[0].mouse_pos)

        # Render Mandelbrot set
        if self.render_mode_id == 0 or self.render_mode_id == 1:
            if self.canvas_list[0].needs_update:
                num_iter = self.fractal_config['MANDELBROT']['N_ITER']
                self._render_fractal_program(self.canvas_list[0], self.program_mandel, self.uniforms_mandel, (0, 0), num_iter)
            self.render_texture(self.window_size, self.canvas_list[0].pos, self.canvas_list[0].size, self.canvas_list[0].framebuffer['COLOR'].tex_id)

        # Render Julia set
        if self.render_mode_id == 1 or self.render_mode_id == 2:
            if self.canvas_list[1].needs_update:
                num_iter = self.fractal_config['JULIA']['N_ITER']
                self._render_fractal_program(self.canvas_list[1], self.program_julia, self.uniforms_julia, self.fractal_constant, num_iter)
            self.render_texture(self.window_size, self.canvas_list[1].pos, self.canvas_list[1].size, self.canvas_list[1].framebuffer['COLOR'].tex_id)

        # Render into text to screen
        if self.text_id == 1 or self.text_id == 2:
            self.render_text(self._get_fractal_info_text(), (10, 8), 1.0, (255, 255, 255))
        if self.text_id == 2 or self.text_id == 3:
            temp_pos_y = self.window_size[1] - 3 * self.render_text.font_size - 8
            self.render_text(self._get_mouse_info_text(), (10, temp_pos_y), 1.0, (255, 255, 255))

        # Update flags
        for canvas in self.canvas_list:
            canvas.needs_update = False

        # Swap buffers and update timings
        glfw.swap_buffers(self.window)
        self.clock.update()




'''
O------------------------------------------------------------------------------O
| AUXILIARY FUNCTIONS                                                          |
O------------------------------------------------------------------------------O
'''

def default_if_none(default_value, value=None):
    output_value = default_value
    if value is not None:
        output_value = value
    if hasattr(output_value, 'copy'):
        output_value = output_value.copy()
    return output_value


def create_main_window(size, vsync=True):
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
    glfw.set_window_size_limits(window, 400, 250, GLFW_VAR.GLFW_DONT_CARE, GLFW_VAR.GLFW_DONT_CARE)
    glfw.make_context_current(window)
    glfw.swap_interval(int(vsync))  # V-sync (refresh rate limit)
    return window


def glfw_get_current_window_monitor(window):
    # Get all available monitors
    monitors = list(glfw.get_monitors())
    num_monitors = len(monitors)
    if num_monitors == 1:
        return monitors[0]
    # Get window bounding box
    window_TL = np.asarray(glfw.get_window_pos(window))
    window_BR = window_TL + np.asarray(glfw.get_window_size(window))
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

