# FractalRendering
Interactive app for visualization of Mandelbrot Set written in Python and GLSL.  

![main_app.py](docs/main_app.gif)


## Features
* Interactive panning and zooming
* Arbitrary window resizing
* Customizable DPI-scaling
* Customizable keymap
* Various colormaps


## Usage
**Note:** Please make sure you have installed the latest version of [Python](https://www.python.org/downloads/) before 
continuing. Copy the following commands into the terminal.   

1. Clone the repository:  
   ```sh  
   git clone git@github.com:MarkoLeskovar/FractalRendering.git 
   cd FractalRendering/
   ```
   
2. Install required python modules via `requirements.txt`:  
   ```sh  
   pip install -r requirements.txt
   ```
   
3. Run the main script to run the interactive app:
   ```sh  
   python main_app.py
   ```


## Customization
Change the `fractals/assets/congif.txt` file to change input keys, various fractal settings and colormap lists.

## Development
- [x] Interactive visualization of the Mandelbrot Set fractal.
- [ ] Create an executable file via PyInstaller.
- [ ] Add Julia Set fractal rendering.

## Links

Mandelbrot Set - [Wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set)  
Plotting algorithms for the Mandelbrot set - [Wikipedia](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set)  
Collection of colormaps by Pratiman Patel - [GitHub.io](https://pratiman-91.github.io/colormaps)  
Collection of font files in .ttf format - [FontSquirrel](https://www.fontsquirrel.com)  

