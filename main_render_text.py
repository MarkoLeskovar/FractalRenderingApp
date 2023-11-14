import glfw
import glfw.GLFW as GLFW_VAR
from OpenGL.GL import *

# Add python modules
from fractals.clock import ClockGLFW
from fractals.text_render import TextRender



# Define main function
def main():

    # Define window size
    window_size = (1600, 1000)

    # Create a GLFW window
    glfw.init()
    glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(GLFW_VAR.GLFW_CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(GLFW_VAR.GLFW_DOUBLEBUFFER, GLFW_VAR.GLFW_TRUE)
    window = glfw.create_window(window_size[0], window_size[1], 'Text rendering', None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Initialize a clock
    clock = ClockGLFW()

    # Initialize a text rendered
    text_renderer = TextRender()
    text_renderer.SetFont(r'C:\Windows\Fonts\arial.ttf', font_size=50)
    text_renderer.SetWindowSize(window_size)


    # Define some text
    test_text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit,\n' \
                'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n' \
                'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris\n' \
                'nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in\n' \
                'reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla\n' \
                'pariatur. Excepteur sint occaecat cupidatat non proident, sunt in\n' \
                'culpa qui officia deserunt mollit anim id est laborum.'

    # Main render loop
    while not glfw.window_should_close(window):

        # Pool events
        glfw.poll_events()
        clock.Update()
        clock.ShowFrameRate(window)

        # Clear the current framebuffer
        glClearColor(0.1, 0.1, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Render text
        text_renderer.DrawText(f'Time = {glfw.get_time():.2f}s', 500, 300, 1.5, (255, 150, 100))
        text_renderer.DrawText(test_text, 0, 0, 0.5, (255, 0, 0))
        text_renderer.DrawText(test_text, 1, 400, 1.0, (0, 255, 0))

        # Swap buffers
        glfw.swap_buffers(window)

    # Terminate the app
    text_renderer.Delete()
    glfw.terminate()


# Run main function
if __name__ == '__main__':
    main()