# Import the necessary PyInstaller hooks
from PyInstaller.utils.hooks import collect_dynamic_libs

# TODO : Check how to run the main script on windows terminal
# TODO : Move all the comments to a seperate develop branch
# TODO : Check this and make everything run
# TODO : Run pyinstaller from a python script. Basically make one click build script for some example function
# TODO : Include a file with assets
# TODO : Add icon file

# Main function call
if __name__ == '__main__':

    # Specify the binaries in the spec file
    glfw_libs = collect_dynamic_libs('glfw')
    freetype_libs = collect_dynamic_libs('freetype')

    pass


