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
    def __init__(self, window_size, font_size=50, x_bounds=(-10, 10), y_bounds=None):
        self.is_running = True
        self.win_size = window_size
        self.font_size = font_size
        self.win_aspect_ratio = self.win_size[0] / self.win_size[1]

        # Initialize the window
        pygame.init()
        self.window = self.create_window(self.win_size)

        # Initialize the clock
        self.clock = pygame.time.Clock()

        # Initialize the font
        self.font = pygame.font.SysFont('freemono', size=self.font_size)

        # Initialize panning and zooming variable
        self.win_shift_default = np.asarray([0.5 * self.win_size[0], -0.5 * self.win_size[1]], dtype='float')
        self.win_shift = self.win_shift_default.copy()
        self.win_scale_default = 1.0
        self.win_scale = self.win_scale_default
        self.win_scale_max = 100.0
        self.win_scale_min = 0.5
        self.win_scale_step = 1.01

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

#
    def create_window(self, win_size):

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


    # Transform from SCREEN space to WORLD space
    def s2w(self, points):
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = (points[0] - self.win_shift[0]) / self.win_scale
        output_points[1] = (self.win_size[1] + self.win_shift[1] - points[1]) / self.win_scale
        return output_points


    # Transform from WORLD space to SCREEN space
    def w2s(self, points):
        output_points = np.empty(points.shape, dtype='float')
        output_points[0] = self.win_shift[0] + points[0] * self.win_scale
        output_points[1] = self.win_size[1] + self.win_shift[1] - points[1] * self.win_scale
        return output_points


    def update_on_resize(self):
        self.win_size = pygame.display.get_window_size()
        self.win_aspect_ratio = self.win_size[0] / self.win_size[1]
        self.surf_background = pygame.Surface(self.win_size)
        self.surf_foreground = pygame.Surface(self.win_size, pygame.SRCALPHA, 32).convert_alpha()
        # Update window shift
        self.win_shift_default = np.asarray([0.5 * self.win_size[0], -0.5 * self.win_size[1]])
        self.win_shift = self.win_shift_default.copy()


    def update_pan_and_zoom(self):

        # Get the mouse pointer
        self.mp_s = np.asarray(pygame.mouse.get_pos())
        self.mp_w = self.s2w(self.mp_s)
        # Pan the screen

        if (self.pressed_mouse_keys[1] == True):
            self.win_shift += self.mp_s - self.mp_s_previous
        # Zoom-in the screen

        if (self.pressed_keys[pygame.K_UP]):
            MP_temp_start = self.s2w(self.mp_s)  # Starting position for the mouse
            self.win_scale *= self.win_scale_step  # Scale also changes "s2w" and "w2s" functions
            if self.win_scale > self.win_scale_max:
                self.win_scale = self.win_scale_max  # Max zoom
            self.win_shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(MP_temp_start)  # Correct position by panning
        # Zoom-out the screen

        if (self.pressed_keys[pygame.K_DOWN]):
            MP_temp_start = self.s2w(self.mp_s)  # Starting position for the mouse
            self.win_scale *= 1.0 / self.win_scale_step  # Scale also changes "s2w" and "w2s" functions
            if self.win_scale < self.win_scale_min:
                self.win_scale = self.win_scale_min  # Min zoom
            self.win_shift += self.w2s(self.s2w(self.mp_s)) - self.w2s(MP_temp_start)  # Correct position by panning

        # Save the current mouse position for the next frame
        self.mp_s_previous = self.mp_s

        # Reset pan and zoom
        if (self.pressed_keys[pygame.K_r]):
            self.win_scale = self.win_scale_default
            self.win_shift = self.win_shift_default.copy()

    def show_window_info(self, surface, color):

        # Show frames per second
        temp_text = f'FPS : {int(self.clock.get_fps())}'
        surface.blit(self.font.render(temp_text, False, color), (0, 0))

        # Show window resolution
        temp_text = f'WINDOW : {self.win_size[0]} x {self.win_size[1]}'
        surface.blit(self.font.render(temp_text, False, color), (0, self.font_size))

        # Show scaling factor
        temp_text = f'ZOOM : {self.win_scale:.2f}'
        surface.blit(self.font.render(temp_text, False, color), (0, 2 * self.font_size))


    def render_loop(self):

        # Create a test surface
        test_image = pygame.image.load('assets/cat.png').convert_alpha()

        # Main render loop
        while self.is_running:

            # Check for events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                if event.type == pygame.WINDOWRESIZED:
                    self.update_on_resize()

            # Grab pressed keys
            self.pressed_mouse_keys = pygame.mouse.get_pressed(3)
            self.pressed_keys = pygame.key.get_pressed()

            # Panning and zooming
            self.update_pan_and_zoom()

            # Fill surfaces
            self.surf_background.fill('black')
            self.surf_foreground.fill((0, 0, 0, 0))

            # Display screen info
            self.show_window_info(self.surf_foreground, pygame.color.THECOLORS['white'])


            # DEBUG - Draw a line set
            line_points = np.asarray([[0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0], [0.0, 0.0]])
            for i in range(line_points.shape[0] - 1):
                temp_start = self.w2s(line_points[i, :])
                temp_end = self.w2s(line_points[i + 1, :])
                pygame.draw.line(self.surf_background, pygame.color.THECOLORS['yellow'], temp_start, temp_end, 2)

            # DEBUG - Show a scaled image
            img_origin = np.asarray([100, 200])
            img_size = np.asarray([200, 200]) * self.win_scale
            img = pygame.transform.scale(test_image, img_size)
            self.surf_background.blit(img, self.w2s(img_origin))


            # Blit surfaces
            self.window.blit(self.surf_background, (0, 0))
            self.window.blit(self.surf_foreground, (0, 0))

            # Update clock and screen
            self.clock.tick()
            pygame.display.flip()



if __name__ == "__main__":
    app = MainApp(window_size=(1600, 900), font_size=50, x_bounds=(-10, 10), y_bounds=None)
