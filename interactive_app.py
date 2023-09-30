import os
import sys
import time
import ctypes
import pygame
import numpy as np

# Import python modules
from mandelbrot import ColorFractal
from mandelbrot import ComputeMandelbrotSet

# Make the application aware of the DPI scaling
# if sys.platform == 'win32':
#    ctypes.windll.user32.SetProcessDPIAware()


# TODO : 01. Hide coordinate axis markings if not in the visible region
# TODO : Add a display to show the number of iterations
# TODO : Add functionality so that mandelbrot only updates when necessary
# TODO : Add functionality to save a screenshot photo with metadata

class MainApp:

    def __init__(self, window_size=(1200, 800), range_x=(-2.5, 1.5), font_size=50):
        self.win_size = tuple(window_size)
        self.range_x_default = tuple(range_x)
        self.font_size = font_size

        # Initialize toggle flags
        self.show_info = True
        self.show_axis = True
        self.is_running = True
        self.needs_updating = True

        # Create a window
        pygame.init()
        self.window = self.create_window(self.win_size)

        # Initialize the font
        self.font = pygame.font.SysFont('freemono', size=self.font_size)

        # Initialize number of iteration variables
        self.num_iter = 200
        self.num_iter_min = 50
        self.num_iter_max = 2000
        self.num_iter_step = 50

        # Initialize shift and scale variables
        self.scale_min = 0.5
        self.scale_max = 1.0e16
        self.scale_step = 0.01
        self.shift_default, self.scale_default = self.compute_shift_and_scale(self.range_x_default, (0.0, 0.0), self.win_size)
        self.shift = self.shift_default.copy()
        self.scale = self.scale_default

        # Initialize the mouse pointer variables
        self.mp_s = np.asarray([0, 0], dtype='uint')
        self.mp_s_previous = self.mp_s.copy()

        # Initialize drawing surface
        self.foreground = pygame.Surface(self.win_size, pygame.SRCALPHA, 32).convert_alpha()

        # Run the main render loop
        self.render_time = 0.0
        self.render_loop()

        # Terminate the app
        pygame.display.quit()
        pygame.quit()
        sys.exit()


    def create_window(self, win_size):
        win_size = list(win_size)

        # Clamp the window dimensions
        min_size = [200, 200]
        if win_size[0] < min_size[0]:
            win_size[0] = min_size[0]
        if win_size[1] < min_size[1]:
            win_size[1] = min_size[1]

        # Create a pygame window
        temp_flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        window = pygame.display.set_mode(win_size, temp_flags)
        pygame.display.set_caption('Mandelbrot Set Rendering')

        # Set the window icon
        icon = pygame.image.load('assets/mandelbrot.png').convert_alpha()
        icon = pygame.transform.scale(icon, (32, 32))
        pygame.display.set_icon(icon)

        # Return the window
        return window


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
        range_x = [TL_w[0], BR_W[0]]
        range_y = [TL_w[1], BR_W[1]]
        return range_x, range_y


    # O------------------------------------------------------------------------------O
    # | PROCESS EVENTS                                                               |
    # O------------------------------------------------------------------------------O

    def process_events(self):

        # Set the current mouse position
        self.mp_s = np.asarray(pygame.mouse.get_pos()).astype('int')

        # Get pygame events and pressed keys
        pygame_events = pygame.event.get()
        pressed_mouse_keys = pygame.mouse.get_pressed(3)
        pressed_keyboard_keys = pygame.key.get_pressed()

        # Process events
        for event in pygame_events:
            # Quit the app
            if event.type == pygame.QUIT:
                self.is_running = False
            # Resize window
            if event.type == pygame.WINDOWSIZECHANGED:
                self.window_resize()
            # Pressed down keys
            if event.type == pygame.KEYDOWN:
                # Quit the app
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                # Toggle info
                if event.key == pygame.K_i:
                    self.show_info = not self.show_info
                # Toggle axis
                if event.key == pygame.K_a:
                    self.show_axis = not self.show_axis
                # Increase number of iterations
                if event.key == pygame.K_KP_PLUS:
                    self.iterations_increase()
                # Decrease number of iterations
                if event.key == pygame.K_KP_MINUS:
                    self.iterations_decrease()
                # Reset shift and scale
                if event.key == pygame.K_r:
                    self.window_reset_shift_and_scale()
            # Mouse wheel movement
            if event.type == pygame.MOUSEWHEEL:
                if event.y == 1:
                    self.window_zoom_in(2.0 * self.scale_step)
                if event.y == -1:
                    self.window_zoom_out(2.0 * self.scale_step)

        # Process shift and scale
        if (pressed_mouse_keys[0] == True):
            self.window_shift()
        if (pressed_keyboard_keys[pygame.K_UP]):
            self.window_zoom_in(self.scale_step)
        if (pressed_keyboard_keys[pygame.K_DOWN]):
            self.window_zoom_out(self.scale_step)

        # Update previous mouse pointer position
        self.mp_s_previous = self.mp_s


    # O------------------------------------------------------------------------------O
    # | DEFINE EVENTS                                                                |
    # O------------------------------------------------------------------------------O

    def window_resize(self):
        self.needs_updating = True
        # Get current range of x-axis
        range_x, range_y = self.get_window_range()
        # Update variables
        self.win_size = pygame.display.get_window_size()
        self.window = self.create_window(self.win_size)
        self.foreground = pygame.Surface(self.win_size, pygame.SRCALPHA, 32).convert_alpha()
        # Update pan and shift
        self.shift_default, self.scale_default = self.compute_shift_and_scale(self.range_x_default, (0.0, 0.0), self.win_size)
        self.shift, self.scale = self.compute_shift_and_scale(range_x, range_y, self.win_size)


    def window_shift(self):
        delta_shift = self.mp_s - self.mp_s_previous
        if (delta_shift[0] != 0.0) or (delta_shift[0] != 0.0):
            self.needs_updating = True
            self.shift += delta_shift


    def window_zoom_in(self, scale_step):
        self.needs_updating = True
        temp_MP_w_start = self.s2w(self.mp_s)  # Starting position for the mouse
        self.scale *= (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self.scale / self.scale_default) > self.scale_max:
            self.scale = self.scale_max * self.scale_default  # Max zoom
        self.shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(temp_MP_w_start)  # Correct position by panning


    def window_zoom_out(self, scale_step):
        self.needs_updating = True
        temp_MP_w_start = self.s2w(self.mp_s)  # Starting position for the mouse
        self.scale *= 1.0 / (1.0 + scale_step)  # Scale also changes "s2w" and "w2s" functions
        if (self.scale / self.scale_default) < self.scale_min:
            self.scale = self.scale_min * self.scale_default  # Min zoom
        self.shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(temp_MP_w_start)  # Correct position by panning


    def window_reset_shift_and_scale(self):
        self.needs_updating = True
        self.shift = self.shift_default.copy()
        self.scale = self.scale_default


    def iterations_increase(self):
        self.needs_updating = True
        self.num_iter += self.num_iter_step
        if self.num_iter > self.num_iter_max:
            self.num_iter = self.num_iter_max


    def iterations_decrease(self):
        self.needs_updating = True
        self.num_iter -= self.num_iter_step
        if self.num_iter < self.num_iter_min:
            self.num_iter = self.num_iter_min


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
    # | SHOW WINDOW INFORMATION                                                      |
    # O------------------------------------------------------------------------------O

    def show_window_info(self, surface, color):

        # Show X and Y coordinates
        temp_pointer = self.s2w(np.asarray(pygame.mouse.get_pos()))
        temp_text_x = f'REAL : {temp_pointer[0]}'
        temp_text_y = f'IMAG : {temp_pointer[1]}'
        surface.blit(self.font.render(temp_text_x, False, color), (5, 0))
        surface.blit(self.font.render(temp_text_y, False, color), (5, self.font_size))

        # Show window resolution
        temp_text = f'SIZE : {self.win_size[0]}x{self.win_size[1]}'
        surface.blit(self.font.render(temp_text, False, color), (5, 2 * self.font_size))

        # Show scaling factor
        temp_scale = self.scale / self.scale_default
        temp_text = f'ZOOM : {temp_scale:.2E}'
        surface.blit(self.font.render(temp_text, False, color), (5, 3 * self.font_size))

        # Show max number of iterations
        temp_text = f'ITER : {self.num_iter}'
        surface.blit(self.font.render(temp_text, False, color), (5, 4 * self.font_size))

        # Show computation time per second
        temp_text = f'TIME : {self.render_time:.4f}'
        surface.blit(self.font.render(temp_text, False, color), (5, 5 * self.font_size))


    def show_coordinate_axis(self, surface, color):

        # Draw coordinate axis
        center_point = self.w2s(np.asarray([0.0, 0.0]))
        pygame.draw.line(surface, color, (center_point[0], 0), (center_point[0], self.win_size[1]), width=1)
        pygame.draw.line(surface, color, (0, center_point[1]), (self.win_size[0], center_point[1]), width=1)

        # Render axis markings
        text_x_max = self.font.render('+Re', False, color)
        text_x_min = self.font.render('-Re', False, color)
        text_y_max = self.font.render('+Im', False, color)
        text_y_min = self.font.render('-Im', False, color)

        # Blit the surfaces
        surface.blit(text_x_max, (self.win_size[0] - text_x_max.get_size()[0], center_point[1]))
        surface.blit(text_x_min, (0, center_point[1]))
        surface.blit(text_y_max, (center_point[0], 0))
        surface.blit(text_y_min, (center_point[0], self.win_size[1] - text_y_min.get_size()[1]))


    def render_loop(self):

        # Initialize the image
        range_x, range_y = self.get_window_range()
        img_fractal_iter = ComputeMandelbrotSet(np.asarray(range_x), np.asarray(range_y), np.asarray(self.win_size), self.num_iter)
        img_fractal_color = ColorFractal(img_fractal_iter.transpose())
        surf_fractal = pygame.surfarray.make_surface(img_fractal_color)

        # Create a test surface
        refresh_icon = pygame.image.load('assets/refresh_icon.png').convert_alpha()
        refresh_icon = pygame.transform.smoothscale(refresh_icon, (self.font_size, self.font_size))

        # Main render loop
        while self.is_running:

            # Process events
            self.process_events()

            # Fill surfaces
            self.window.fill('black')
            self.foreground.fill((0, 0, 0, 0))

            # Display information and coordinate axis
            if self.show_info:
                self.show_window_info(self.foreground, pygame.color.THECOLORS['white'])
            if self.show_axis:
                self.show_coordinate_axis(self.foreground, pygame.color.THECOLORS['white'])
            if self.show_info and self.needs_updating:
                self.foreground.blit(refresh_icon, (self.win_size[0] - self.font_size, 0))

            # Compute the image
            # t0 = time.time()
            # range_x, range_y = self.get_window_range()
            # img_fractal_iter = ComputeMandelbrotSet(np.asarray(range_x), np.asarray(range_y), np.asarray(self.win_size), self.num_iter)
            # img_fractal_color = ColorFractal(img_fractal_iter.transpose())
            # surf_fractal = pygame.surfarray.make_surface(img_fractal_color)
            # self.render_time = time.time() - t0

            # DEBUG - Draw a line set
            line_points = np.asarray([[-2.5, -1.5], [-2.5, 1.5], [1.5, 1.5], [1.5, -1.5], [-2.5, -1.5]])
            for i in range(line_points.shape[0] - 1):
                temp_start = self.w2s(line_points[i, :])
                temp_end = self.w2s(line_points[i + 1, :])
                pygame.draw.line(self.window, pygame.color.THECOLORS['red'], temp_start, temp_end, 2)

            # Change updating flag
            self.needs_updating = False

            # Blit surfaces and flip display
            # self.window.blit(surf_fractal, (0, 0))
            self.window.blit(self.foreground, (0, 0))
            pygame.display.flip()



if __name__ == "__main__":
    app = MainApp(window_size=(600, 400), font_size=20)
