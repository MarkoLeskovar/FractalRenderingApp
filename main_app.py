# Import the main app
from fractals.fractal_render import FractalRenderingApp

# Main function call
if __name__ == '__main__':
    app = FractalRenderingApp(window_size=(800, 450),
        config_file='fractals/assets/config.txt',
    )
    app.Run()
    app.Close()
