import sys
import ctypes
import pygame
import numpy as np
import tkinter as tk

# Make the application aware of the DPI scaling
if sys.platform == 'win32':
   ctypes.windll.user32.SetProcessDPIAware()

from mandelbrot import ComputeMandelbrotSet
# TODO : Add functionality so that scale doesn't reset on windows resize

class MainApp:

    def __init__(self, window_size=(1200, 800), font_size=50):
        self.is_running = True
        self.win_size = window_size
        self.font_size = font_size

        # Init toggle flags
        self.show_info = True
        self.show_axis = True
        self.needs_updating = True

        # Create a window
        pygame.init()
        self.window = self.create_window(self.win_size)

        # Initialize the clock
        self.clock = pygame.time.Clock()

        # Initialize the font
        self.font = pygame.font.SysFont('freemono', size=self.font_size)

        # Initialize settings
        self.max_iter = 200
        self.x_bounds = np.asarray([-2.5, 1.5])

        # Initialize panning and zooming
        self.init_pan_and_zoom()

        # Initialize drawing surfaces
        self.surf_background = pygame.Surface(self.win_size)
        self.surf_foreground = pygame.Surface(self.win_size, pygame.SRCALPHA, 32).convert_alpha()

        # Run the render loop
        self.render_loop()

        # Terminate the app
        pygame.quit()
        sys.exit()


    def create_window(self, win_size):
        win_size = list(win_size)

        # Clamp the window dimensions
        min_size = (200, 200)
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


    def init_pan_and_zoom(self):

        # Compute image dimensions
        x_size = self.x_bounds[1] - self.x_bounds[0]
        self.pix_size = x_size / self.win_size[0]

        # Initialize scaling variable
        self.scale_min = 0.5
        self.scale_max = 1.0e6
        self.scale_step = 1.01
        self.scale_default = 1.0 / self.pix_size
        self.scale = self.scale_default

        # Initialize panning variables
        temp_shift_x = 0.5 * self.win_size[0]  # Offset by image center
        temp_shift_x -= (self.x_bounds[0] + 0.5 * x_size) * self.scale_default  # Offset by bounds
        temp_shift_y = -0.5 * self.win_size[1]  # Offset by image center
        self.shift_default = np.asarray([temp_shift_x, temp_shift_y], dtype='float')
        self.shift = self.shift_default.copy()

        # Initialize the mouse pointer variables
        self.mp_s_previous = np.asarray([0, 0], dtype='uint')


    def process_events(self):
        for event in pygame.event.get():
            # Quit the app
            if event.type == pygame.QUIT:
                self.is_running = False
            # Resize window
            if event.type == pygame.WINDOWSIZECHANGED:
                self.needs_updating = True
                self.win_size = pygame.display.get_window_size()
                self.window = self.create_window(self.win_size)
                self.surf_background = pygame.Surface(self.win_size)
                self.surf_foreground = pygame.Surface(self.win_size, pygame.SRCALPHA, 32).convert_alpha()
                self.init_pan_and_zoom()
            # Key strokes
            if event.type == pygame.KEYDOWN:
                # Toggle info
                if event.key == pygame.K_i:
                    self.show_info = not self.show_info
                # Toggle axis
                if event.key == pygame.K_a:
                    self.show_axis = not self.show_axis
                # Increase max iterations
                if event.key == pygame.K_KP_PLUS:
                    self.max_iter += 10
                    if self.max_iter > 1000:
                        self.max_iter = 1000
                # Decrease max iterations
                if event.key == pygame.K_KP_MINUS:
                    self.max_iter -= 10
                    if self.max_iter < 10:
                        self.max_iter = 10


    def process_pan_and_zoom(self):

        # Get pressed keys
        pressed_mouse_keys = pygame.mouse.get_pressed(3)
        pressed_keys = pygame.key.get_pressed()

        # Get the mouse pointer position
        self.mp_s = np.asarray(pygame.mouse.get_pos())

        # Pan the screen
        if (pressed_mouse_keys[1] == True):
            delta_shift = self.mp_s - self.mp_s_previous
            if (delta_shift[0] != 0.0) or (delta_shift[0] != 0.0):
                self.needs_updating = True
                self.shift += delta_shift

        # Zoom-in the screen
        if (pressed_keys[pygame.K_UP]):
            self.needs_updating = True
            MP_temp_start = self.s2w(self.mp_s)  # Starting position for the mouse
            self.scale *= self.scale_step  # Scale also changes "s2w" and "w2s" functions
            if (self.scale / self.scale_default) > self.scale_max:
                self.scale = self.scale_max * self.scale_default  # Max zoom
            self.shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(MP_temp_start)  # Correct position by panning

        # Zoom-out the screen
        if (pressed_keys[pygame.K_DOWN]):
            self.needs_updating = True
            MP_temp_start = self.s2w(self.mp_s)  # Starting position for the mouse
            self.scale *= 1.0 / self.scale_step  # Scale also changes "s2w" and "w2s" functions
            if (self.scale / self.scale_default) < self.scale_min:
                self.scale = self.scale_min * self.scale_default  # Min zoom
            self.shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(MP_temp_start)  # Correct position by panning

        # Save the current mouse position for the next frame
        self.mp_s_previous = self.mp_s

        # Reset pan and zoom
        if (pressed_keys[pygame.K_r]):
            self.needs_updating = True
            self.scale = self.scale_default
            self.shift = self.shift_default.copy()


    # O------------------------------------------------------------------------------O
    # | SCREEN TO WORLD & WORLD TO SCREEN TRANSFORMATIONS                            |
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

        # Show frames per second
        temp_text = f'FPS : {int(self.clock.get_fps())}'
        surface.blit(self.font.render(temp_text, False, color), (0, 0))

        # Show window resolution
        temp_text = f'SIZE : {self.win_size[0]} x {self.win_size[1]}'
        surface.blit(self.font.render(temp_text, False, color), (0, self.font_size))

        # Show scaling factor
        temp_scale = self.scale / self.scale_default
        temp_text = f'ZOOM : {temp_scale:.2f}'
        surface.blit(self.font.render(temp_text, False, color), (0, 2 * self.font_size))

        # Show number of iterations
        temp_text = f'ITER : {self.max_iter}'
        surface.blit(self.font.render(temp_text, False, color), (0, 3 * self.font_size))


    def show_coordinate_axis(self, surface, color):

        # Draw coordinate axis
        center_point = self.w2s(np.asarray([0.0, 0.0]))
        pygame.draw.line(surface, color, (center_point[0], 0), (center_point[0], self.win_size[1]), width=1)
        pygame.draw.line(surface, color, (0, center_point[1]), (self.win_size[0], center_point[1]), width=1)

        # Render axis markings
        text_x_max = self.font.render('+X', False, color)
        text_x_min = self.font.render('-X', False, color)
        text_y_max = self.font.render('+Y', False, color)
        text_y_min = self.font.render('-Y', False, color)

        # Blit the surfaces
        surface.blit(text_x_max, (self.win_size[0] - text_x_max.get_size()[0], center_point[1]))
        surface.blit(text_x_min, (0, center_point[1]))
        surface.blit(text_y_max, (center_point[0], 0))
        surface.blit(text_y_min, (center_point[0], self.win_size[1] - text_y_min.get_size()[1]))


    def render_loop(self):

        # Create a test surface
        test_image = pygame.image.load('assets/cat.png').convert_alpha()
        test_image = pygame.transform.scale(test_image, (200, 200))

        # Image extent
        TL_s = np.asarray([0, 0])
        BR_s = np.asarray([self.win_size[0], self.win_size[1]])

        # Main render loop
        while self.is_running:

            # Process events
            self.process_events()
            self.process_pan_and_zoom()

            # Fill surfaces
            self.surf_background.fill('black')
            self.surf_foreground.fill((0, 0, 0, 0))

            # Display information and coordinate axis
            if self.show_info:
                self.show_window_info(self.surf_foreground, pygame.color.THECOLORS['white'])
            if self.show_axis:
                self.show_coordinate_axis(self.surf_foreground, pygame.color.THECOLORS['green'])

            # Get bounds
            TL_w = self.s2w(TL_s)
            BR_w = self.s2w(BR_s)
            x_bounds = np.asarray([TL_w[0], BR_w[0]])
            y_bounds = np.asarray([TL_w[1], BR_w[1]])

            # TODO : Implement some coloring scheme

            # Compute the set
            # iterations = ComputeMandelbrotSet(x_bounds, y_bounds, np.asarray(self.win_size), self.max_iter)

            # DEBUG - Draw a line set
            line_points = np.asarray([[-2.5, -1.5], [-2.5, 1.5], [1.5, 1.5], [1.5, -1.5], [-2.5, -1.5]])
            for i in range(line_points.shape[0] - 1):
                temp_start = self.w2s(line_points[i, :])
                temp_end = self.w2s(line_points[i + 1, :])
                pygame.draw.line(self.surf_background, pygame.color.THECOLORS['yellow'], temp_start, temp_end, 2)

            if self.needs_updating:
                self.surf_background.blit(test_image, (0, 0))


            # Change updating flag
            self.needs_updating = False

            # Blit surfaces
            self.window.blit(self.surf_background, (0, 0))
            self.window.blit(self.surf_foreground, (0, 0))

            # Update clock and screen
            self.clock.tick()
            pygame.display.flip()



if __name__ == "__main__":
    app = MainApp(window_size=(1600, 1200), font_size=50)
