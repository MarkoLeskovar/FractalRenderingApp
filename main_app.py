from fractals.interactive_app import FractalRenderingApp

# Main function call
if __name__ == '__main__':
    app = FractalRenderingApp(
        cmap_file='fractals/assets/cmaps_diverging.txt',
        keymap_file='fractals/assets/keymap.txt',
    )
