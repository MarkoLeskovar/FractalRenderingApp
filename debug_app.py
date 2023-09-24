import sys
import ctypes
import pygame
import numpy as np
import tkinter as tk

# Make the application aware of the DPI scaling
if sys.platform == 'win32':
   ctypes.windll.user32.SetProcessDPIAware()


# TODO : Implement bounds for some default scaling!
# TODO : Enable picture scaling with some bounds, so that I do not lose FPS!

class MainApp:
    def __init__(self, window_size, font_size=50):
        self.is_running = True
        self.win_size = window_size
        self.font_size = font_size

        # Initialize the window
        pygame.init()
        self.window = self.create_window(self.win_size)

        # Initialize the clock
        self.clock = pygame.time.Clock()

        # Initialize the font
        self.font = pygame.font.SysFont('freemono', size=self.font_size)

        # Initialize panning and zooming
        self.x_bounds = np.asarray([-2.5, 1.5])
        self.init_pan_and_zoom()

        # Initialize the mouse pointer variables
        self.mp_s_previous = np.asarray([0, 0], dtype='uint')
        self.mp_w_previous = np.asarray([0.0, 0.0], dtype='float')
        self.mp_s = np.asarray([0.0, 0.0], dtype='uint')
        self.mp_w = self.s2w(self.mp_s)

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

        # Clamp window to minimal size
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


    # Transform from SCREEN space to WORLD space
    def s2w(self, points):
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = (points[0] - self.shift[0]) / self.scale
        output_points[1] = (self.win_size[1] + self.shift[1] - points[1]) / self.scale
        return output_points


    # Transform from WORLD space to SCREEN space
    def w2s(self, points):
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = self.shift[0] + points[0] * self.scale
        output_points[1] = self.win_size[1] + self.shift[1] - points[1] * self.scale
        return output_points


    def update_on_resize(self):
        self.win_size = pygame.display.get_window_size()
        self.window = self.create_window(self.win_size)
        self.surf_background = pygame.Surface(self.win_size)
        self.surf_foreground = pygame.Surface(self.win_size, pygame.SRCALPHA, 32).convert_alpha()
        self.init_pan_and_zoom()


    def update_pan_and_zoom(self):

        # Get the mouse pointer
        self.mp_s = np.asarray(pygame.mouse.get_pos())
        self.mp_w = self.s2w(self.mp_s)

        # Pan the screen
        if (self.pressed_mouse_keys[1] == True):
            self.shift += self.mp_s - self.mp_s_previous

        # Zoom-in the screen
        if (self.pressed_keys[pygame.K_UP]):
            MP_temp_start = self.s2w(self.mp_s)  # Starting position for the mouse
            self.scale *= self.scale_step  # Scale also changes "s2w" and "w2s" functions
            if (self.scale / self.scale_default) > self.scale_max:
                self.scale = self.scale_max * self.scale_default  # Max zoom
            self.shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(MP_temp_start)  # Correct position by panning

        # Zoom-out the screen
        if (self.pressed_keys[pygame.K_DOWN]):
            MP_temp_start = self.s2w(self.mp_s)  # Starting position for the mouse
            self.scale *= 1.0 / self.scale_step  # Scale also changes "s2w" and "w2s" functions
            if (self.scale / self.scale_default) < self.scale_min:
                self.scale = self.scale_min * self.scale_default  # Min zoom
            self.shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(MP_temp_start)  # Correct position by panning

        # Save the current mouse position for the next frame
        self.mp_s_previous = self.mp_s

        # Reset pan and zoom
        if (self.pressed_keys[pygame.K_r]):
            self.scale = self.scale_default
            self.shift = self.shift_default.copy()


    # O------------------------------------------------------------------------------O
    # | PRIVATE - SHOW INFORMATION                                                   |
    # O------------------------------------------------------------------------------O

    def show_window_info(self, surface, color):

        # Show frames per second
        temp_text = f'FPS : {int(self.clock.get_fps())}'
        surface.blit(self.font.render(temp_text, False, color), (0, 0))

        # Show window resolution
        temp_text = f'WINDOW : {self.win_size[0]} x {self.win_size[1]}'
        surface.blit(self.font.render(temp_text, False, color), (0, self.font_size))

        # Show scaling factor
        temp_scale = self.scale / self.scale_default
        temp_text = f'ZOOM : {temp_scale:.2f}'
        surface.blit(self.font.render(temp_text, False, color), (0, 2 * self.font_size))


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
        surface.blit(text_x_max, (center_point[0], 0))
        surface.blit(text_x_min, (center_point[0], self.win_size[1] - text_x_min.get_size()[1]))
        surface.blit(text_y_max, (self.win_size[0] - text_y_max.get_size()[0], center_point[1]))
        surface.blit(text_y_min, (0, center_point[1]))


    def render_loop(self):

        show_info = True
        show_axis = True


        # Create a test surface
        # test_image = pygame.image.load('assets/cat.png').convert_alpha()

        # Main render loop
        while self.is_running:

            # Check for events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                if event.type == pygame.WINDOWSIZECHANGED:
                    self.update_on_resize()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_i:  # Toggle info
                        show_info = not show_info
                    if event.key == pygame.K_a:  # Toggle axis
                        show_axis = not show_axis

            # Grab pressed keys
            self.pressed_mouse_keys = pygame.mouse.get_pressed(3)
            self.pressed_keys = pygame.key.get_pressed()

            # Panning and zooming
            self.update_pan_and_zoom()

            # Fill surfaces
            self.surf_background.fill('black')
            self.surf_foreground.fill((0, 0, 0, 0))

            # Display information and coordinate axis
            if show_info:
                self.show_window_info(self.surf_foreground, pygame.color.THECOLORS['white'])
            if show_axis:
                self.show_coordinate_axis(self.surf_foreground, pygame.color.THECOLORS['green'])


            # DEBUG - Print extent
            # top_left = self.s2w(np.asarray([0.0, 0.0]))
            # bottom_right = self.s2w(np.asarray([self.win_size[0], self.win_size[1]]))
            # print(top_left)
            # print(bottom_right)


            # DEBUG - Draw a line set
            line_points = np.asarray([[-2.5, -1.5], [-2.5, 1.5], [1.5, 1.5], [1.5, -1.5], [-2.5, -1.5]])
            for i in range(line_points.shape[0] - 1):
                temp_start = self.w2s(line_points[i, :])
                temp_end = self.w2s(line_points[i + 1, :])
                pygame.draw.line(self.surf_background, pygame.color.THECOLORS['yellow'], temp_start, temp_end, 2)


            # Blit surfaces
            self.window.blit(self.surf_background, (0, 0))
            self.window.blit(self.surf_foreground, (0, 0))

            # Update clock and screen
            self.clock.tick()
            pygame.display.flip()



if __name__ == "__main__":
    app = MainApp(window_size=(1600, 1200), font_size=50)
