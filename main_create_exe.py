# Import the necessary PyInstaller hooks
from PyInstaller.utils.hooks import collect_dynamic_libs

# Main function call
if __name__ == '__main__':

    # Specify the binaries in the spec file
    glfw_libs = collect_dynamic_libs('glfw')
    freetype_libs = collect_dynamic_libs('freetype')

    pass


