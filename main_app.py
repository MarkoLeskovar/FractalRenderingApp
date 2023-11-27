from fractals.fractal_render import FractalRenderingApp

# Main function call
if __name__ == '__main__':
    app = FractalRenderingApp(
        config_file='fractals/assets/config.txt',
    )
    app.Run()
    app.Close()
