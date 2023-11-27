# Import the main app
from fractals.fractal_render import FractalRenderingApp

# Main function call
if __name__ == '__main__':
    app = FractalRenderingApp(
        cmap_file='fractals/assets/cmaps_diverging.txt',
        settings_file='fractals/assets/settings.txt',
    )
    app.Run()
    app.Close()
