import os
import shutil
import platform
import subprocess

# Main function call
if __name__ == '__main__':

    # Define paths
    cwd = os.path.dirname(__file__)
    dist_path = os.path.join(cwd, 'dist', 'main_app')

    # Run PyInstaller through the terminal
    print(f'Running Pyinstaller...')
    subprocess.run('pyinstaller main_app.spec --noconfirm', shell=True, cwd=cwd)
    print(f'...done!\n')

    # Zip the files
    print(f'Compressing the "main_app" folder...')
    shutil.make_archive(os.path.join(cwd, 'FractalRenderingApp'), 'zip', dist_path, verbose=True)
    print(f'...done!\n')

    # Delete the build directory
    print(f'Removing "build" and "dist" folders...')
    shutil.rmtree(os.path.join(cwd, 'build'))
    shutil.rmtree(os.path.join(cwd, 'dist'))
    print(f'...done!\n')
