import os
import json
from fractals.color import LoadColormapsFile
from fractals.fractal_render import FractalRenderingApp

PATH_TO_ASSETS = os.path.join(os.path.dirname(__file__), 'fractals', 'assets')

# Main function call
if __name__ == '__main__':

    # Load config files
    with open(os.path.join(PATH_TO_ASSETS, 'config.json')) as f:
        config_file = json.load(f)

    # Load colormaps files
    cmaps_list = LoadColormapsFile(os.path.join(PATH_TO_ASSETS, config_file['CMAPS_FILE']))

    # Change the default path to assets
    FractalRenderingApp.SetPathToAssets(PATH_TO_ASSETS)

    # Run the main app
    app = FractalRenderingApp(
        app_config=config_file['APP'],
        controls_config=config_file['CONTROLS'],
        fractal_config=config_file['FRACTAL'],
        cmaps=cmaps_list,
    )
    app.Run()
    app.Close()
