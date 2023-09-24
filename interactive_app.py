import pygame
import numpy as np


# O------------------------------------------------------------------------------O
# | GLOBAL VARIABLES                                                             |
# O------------------------------------------------------------------------------O

# Window dimensions
WIN_SCREEN_WIDTH = 1200
WIN_SCREEN_HEIGHT = 700
WIN_TOOLBAR_WIDTH = 350

# Window edges in screen space
WIN_TL_S = np.asarray([0, 0], 'uint')
WIN_TR_S = np.asarray([WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH, 0], 'uint')
WIN_BL_S = np.asarray([0, WIN_SCREEN_HEIGHT], 'uint')
WIN_BR_S = np.asarray([WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH, WIN_SCREEN_HEIGHT], 'uint')

# Window midpoints in screen space
WIN_TM_S = np.asarray([0.5 * (WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH), 0], 'uint')
WIN_BM_S = np.asarray([0.5 * (WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH), WIN_SCREEN_HEIGHT], 'uint')
WIN_LM_S = np.asarray([0, 0.5 * WIN_SCREEN_HEIGHT], 'uint')
WIN_RM_S = np.asarray([WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH, 0.5 * WIN_SCREEN_HEIGHT], 'uint')

# Pan and zoom variables
WIN_SHIFT_DEFAULT = np.asarray([0.5 * (WIN_SCREEN_WIDTH + WIN_TOOLBAR_WIDTH) - WIN_TOOLBAR_WIDTH,
                                -0.5 * WIN_SCREEN_HEIGHT], 'float')
WIN_SHIFT = np.copy(WIN_SHIFT_DEFAULT)
WIN_SCALE_DEFAULT = 3.0
WIN_SCALE = WIN_SCALE_DEFAULT
WIN_SCALE_STEP = 1.01
WIN_SCALE_MAX = 10000.0
WIN_SCALE_MIN = 1.0

# Initialize the pygame
pygame.init()
WIN = pygame.display.set_mode((WIN_SCREEN_WIDTH, WIN_SCREEN_HEIGHT))
pygame.display.set_caption("Geometry Visualizer")


# O------------------------------------------------------------------------------O
# | TRANSFORMATIONS BETWEEN SCREEN AND WORLD                                     |
# O------------------------------------------------------------------------------O

# Transform from SCREEN space to WORLD space
def s2w(input_points):
    output_points = np.empty(input_points.shape, dtype='float')
    output_points[0] = (input_points[0] - WIN_SHIFT[0]) / WIN_SCALE
    output_points[1] = (WIN_SCREEN_HEIGHT + WIN_SHIFT[1] - input_points[1]) / WIN_SCALE
    return output_points


# Transform from WORLD space to SCREEN space
def w2s(input_points):
    output_points = np.empty(input_points.shape, dtype='float')
    output_points[0] = WIN_SHIFT[0] + input_points[0] * WIN_SCALE
    output_points[1] = WIN_SCREEN_HEIGHT + WIN_SHIFT[1] - input_points[1] * WIN_SCALE
    return output_points


# Define colors
def DefineAppColors():
    color = {}
    color['selection'] = pygame.color.THECOLORS['yellow']
    color['toolbar'] = pygame.color.THECOLORS['black']
    color['toolbar_selection'] = pygame.color.THECOLORS['green']
    color['background'] = pygame.color.THECOLORS['navyblue']
    color['geometry'] = pygame.color.THECOLORS['white']
    color['toolbar_text'] = pygame.color.THECOLORS['white']
    color['axis'] = pygame.color.THECOLORS['gray']
    return color




# O------------------------------------------------------------------------------O
# | MAIN FUNCTION                                                                |
# O------------------------------------------------------------------------------O
def main():

    # O------------------------------------------------------------------------------O
    # | PREAMBLE                                                                     |
    # O------------------------------------------------------------------------------O

    # Define color
    color = DefineAppColors()

    # Drawing selection
    toolbarMode = False
    showAxis = True

    # Surfaces (layers)
    SURF_background = pygame.Surface((WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH, WIN_SCREEN_HEIGHT))
    SURF_geometry = pygame.Surface((WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH, WIN_SCREEN_HEIGHT), pygame.SRCALPHA, 32)
    SURF_geometry = SURF_geometry.convert_alpha()
    SURF_info = pygame.Surface((WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH, WIN_SCREEN_HEIGHT), pygame.SRCALPHA, 32)
    SURF_info = SURF_info.convert_alpha()
    SURF_axis = pygame.Surface((WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH, WIN_SCREEN_HEIGHT), pygame.SRCALPHA, 32)
    SURF_axis = SURF_axis.convert_alpha()
    SURF_selection = pygame.Surface((WIN_SCREEN_WIDTH - WIN_TOOLBAR_WIDTH, WIN_SCREEN_HEIGHT), pygame.SRCALPHA, 32)
    SURF_selection = SURF_selection.convert_alpha()
    SURF_toolbar = pygame.Surface((WIN_TOOLBAR_WIDTH, WIN_SCREEN_HEIGHT))

    # Fonts
    FONT_toolbar_size = 30
    FONT_toolbar = pygame.font.SysFont('freemono', FONT_toolbar_size)
    FONT_axis_size = 30
    FONT_axis = pygame.font.SysFont('freemono', FONT_axis_size)

    # Clock
    clock = pygame.time.Clock()

    # Initialize the mouse pointer variables
    MP_S_previous = np.asarray([0, 0], dtype='uint')
    MP_W_previous = np.asarray([0.0, 0.0], dtype='float')


    # Main while loop
    running = True
    while running:

        # O------------------------------------------------------------------------------O
        # | PREAMBLE                                                                     |
        # O------------------------------------------------------------------------------O
        global WIN_SHIFT, WIN_SCALE
        clock.tick()

        # Geometry surfaces
        SURF_background.fill(color['background'])
        SURF_geometry.fill((0, 0, 0, 0))
        SURF_axis.fill((0, 0, 0, 0))
        SURF_info.fill((0, 0, 0, 0))
        SURF_selection.fill((0, 0, 0, 0))
        SURF_toolbar.fill(color['toolbar'])

        # Pressed keys
        pressed_mouse_keys = pygame.mouse.get_pressed(3)
        pressed_keys = pygame.key.get_pressed()

        # Check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Geometry
                if event.key == pygame.K_a:  # Toggle axis
                    showAxis = not showAxis


        # O------------------------------------------------------------------------------O
        # | GRAB MOUSE POSITION AND CREATE A BOUNDING BOX                                |
        # O------------------------------------------------------------------------------O
        MP_S = np.asarray(pygame.mouse.get_pos())
        MP_S[0] -= WIN_TOOLBAR_WIDTH
        MP_W = s2w(MP_S)

        # Screen edges
        TL_W = s2w(WIN_TL_S)
        TR_W = s2w(WIN_TR_S)
        BL_W = s2w(WIN_BL_S)
        BR_W = s2w(WIN_BR_S)

        # O------------------------------------------------------------------------------O
        # | PANNING AND ZOOMING                                                          |
        # O------------------------------------------------------------------------------O

        # Pan the screen
        if (not toolbarMode and pressed_mouse_keys[1] == True):
            WIN_SHIFT += MP_S - MP_S_previous
            pygame.draw.circle(SURF_selection, color['selection'], MP_S, radius=10, width=1)

        # Zoom-in the screen
        if (not toolbarMode and pressed_keys[pygame.K_UP]):
            MP_temp_start = s2w(MP_S)  # Starting position for the mouse
            WIN_SCALE *= WIN_SCALE_STEP  # Scale also changes "s2w" and "w2s" functions
            if WIN_SCALE > WIN_SCALE_MAX:
                WIN_SCALE = WIN_SCALE_MAX  # Max zoom
            WIN_SHIFT += w2s(s2w(MP_S)) - w2s(MP_temp_start)  # Correct position by panning
            pygame.draw.circle(SURF_selection, color['selection'], MP_S, radius=10, width=1)

        # Zoom-out the screen
        if (not toolbarMode and pressed_keys[pygame.K_DOWN]):
            MP_temp_start = s2w(MP_S)  # Starting position for the mouse
            WIN_SCALE *= 1.0 / WIN_SCALE_STEP  # Scale also changes "s2w" and "w2s" functions
            if WIN_SCALE < WIN_SCALE_MIN:
                WIN_SCALE = WIN_SCALE_MIN  # Min zoom
            WIN_SHIFT += w2s(s2w(MP_S)) - w2s(MP_temp_start)  # Correct position by panning
            pygame.draw.circle(SURF_selection, color['selection'], MP_S, radius=10, width=1)

        # Check if the position is inside drawing area
        if (MP_S[0] > 0):
            toolbarMode = False
            MP_S_previous = MP_S.copy()
            MP_W_previous = MP_W.copy()
        else:
            toolbarMode = True
            MP_S = MP_S_previous.copy()
            MP_W = MP_W_previous.copy()

        # Save the current mouse position for the next frame
        MP_S_previous = MP_S

        # Reset pan and zoom
        if pressed_keys[pygame.K_r]:
            WIN_SCALE = WIN_SCALE_DEFAULT
            WIN_SHIFT = np.copy(WIN_SHIFT_DEFAULT)

        # O------------------------------------------------------------------------------O
        # | DISPLAY COORDINATE SYSTEM AXIS                                               |
        # O------------------------------------------------------------------------------O
        if showAxis:
            # Draw coordinate axis
            center_point = w2s(np.asarray([0.0, 0.0]))
            pygame.draw.line(SURF_axis, color['axis'], (center_point[0], WIN_TL_S[0]), (center_point[0], WIN_TR_S[0]), width=1)
            pygame.draw.line(SURF_axis, color['axis'], (WIN_TL_S[0], center_point[1]), (WIN_TR_S[0], center_point[1]), width=1)
            # Add axis markings
            SURF_axis.blit(FONT_axis.render('+x', True, color['axis']), (WIN_TR_S[0] - 45, center_point[1]))
            SURF_axis.blit(FONT_axis.render('-x', True, color['axis']), (WIN_TL_S[0] + 5, center_point[1]))
            SURF_axis.blit(FONT_axis.render('+y', True, color['axis']), (center_point[0] + 5, 5))
            SURF_axis.blit(FONT_axis.render('-y', True, color['axis']), (center_point[0] + 5, WIN_BL_S[1] - 35))


        # O------------------------------------------------------------------------------O
        # | DISPLAY SELECTED INTERPOLATION                                               |
        # O------------------------------------------------------------------------------O
        pygame.draw.circle(SURF_selection, color['selection'], MP_S, radius=10, width=1)

        rect_pos = np.asarray([[10.0, 5.0], [30.0, 20.0]])
        pygame.draw.line(SURF_geometry, pygame.color.THECOLORS['white'], w2s(rect_pos[0, :]), w2s(rect_pos[1, :]), width=4)





        # O------------------------------------------------------------------------------O
        # | TOOLBAR INFORMATION                                                          |
        # O------------------------------------------------------------------------------O

        # Add text to the toolbar - FPS
        SURF_toolbar.blit(FONT_toolbar.render(f'FPS = {int(clock.get_fps())}', True, color['toolbar_text']), (5, 5))

        # Divider line
        pygame.draw.line(SURF_toolbar, color['toolbar_text'], (0, 35), (WIN_TOOLBAR_WIDTH, 35), width=1)

        # Add text to the toolbar - X,Y,S
        SURF_toolbar.blit(FONT_toolbar.render(f'X = {MP_W[0]:.3f} mm', True, color['toolbar_text']), (5, 40))
        SURF_toolbar.blit(FONT_toolbar.render(f'Y = {MP_W[1]:.3f} mm', True, color['toolbar_text']), (5, 70))
        SURF_toolbar.blit(FONT_toolbar.render(f'S = {WIN_SCALE:.3f} x', True, color['toolbar_text']), (5, 100))

        # Divider line
        pygame.draw.line(SURF_toolbar, color['toolbar_text'], (0, 130), (WIN_TOOLBAR_WIDTH, 130), width=1)

        # Display options
        if pressed_keys[pygame.K_r]:
            color_reset_view = color['toolbar_selection']
        else:
            color_reset_view = color['toolbar_text']
        if showAxis:
            color_show_axis = color['toolbar_selection']
        else:
            color_show_axis = color['toolbar_text']

        SURF_toolbar.blit(FONT_toolbar.render('[R] - RESET VIEW', True, color_reset_view), (5, 135))
        SURF_toolbar.blit(FONT_toolbar.render('[A] - SHOW AXIS', True, color_show_axis), (5, 165))

        # Divider line
        pygame.draw.line(SURF_toolbar, color['toolbar_text'], (0, 195), (WIN_TOOLBAR_WIDTH, 195), width=1)

        # O------------------------------------------------------------------------------O
        # | MERGE SURFACES                                                               |
        # O------------------------------------------------------------------------------O
        WIN.blit(SURF_background, (WIN_TOOLBAR_WIDTH, 0))
        WIN.blit(SURF_axis, (WIN_TOOLBAR_WIDTH, 0))
        WIN.blit(SURF_geometry, (WIN_TOOLBAR_WIDTH, 0))
        WIN.blit(SURF_selection, (WIN_TOOLBAR_WIDTH, 0))
        WIN.blit(SURF_info, (WIN_TOOLBAR_WIDTH, 0))
        WIN.blit(SURF_toolbar, (0, 0))

        # Update the display
        pygame.display.update()

    # Quit the app
    pygame.quit()


# Main function call
if __name__ == '__main__':
    main()
