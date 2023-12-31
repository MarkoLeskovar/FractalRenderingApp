import os
import sys
import json
from fractals.color import load_colormaps_file
from fractals.render_fractal import FractalRenderingApp

# Define path to assets depending on if the app is bundled or not
PATH_TO_ASSETS = os.path.join(os.path.dirname(__file__), 'fractals', 'assets')
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    PATH_TO_ASSETS = os.curdir


# Main function call
if __name__ == '__main__':

    # Load config files
    with open(os.path.join(PATH_TO_ASSETS, 'config.json')) as f:
        config_file = json.load(f)

    # Load colormaps files
    cmaps_list = load_colormaps_file(os.path.join(PATH_TO_ASSETS, config_file['CMAPS_FILE']))

    # Change the default path to assets
    FractalRenderingApp.set_path_to_assets(PATH_TO_ASSETS)

    # Run the main app
    app = FractalRenderingApp(
        app_config=config_file['APP'],
        controls_config=config_file['CONTROLS'],
        fractal_config=config_file['FRACTAL'],
        cmaps=cmaps_list,
    )
    app.run()
    app.close()
