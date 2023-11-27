# FractalRendering
Interactive app for visualization of Mandelbrot Set written in Python and GLSL.  

![main_app.py](docs/main_app.gif)


## Features

* Interactive panning and zooming
* Arbitrary window resizing
* Automatic DPI-scaling
* 296 available colormaps
* Customizable settings


## How to Use?

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


## Controls & Customization

* Use `ARROW KEY UP` / `DOWN` or `mouse scroll` to zoom in and out.
* Use `ARROW KEY LEFT` / `RIGHT` to change colormaps.
* Use `+` / `-` to increase or decrease the number of fractal iterations.
* Use `*` / `/` to increase or decrease the pixel scaling.
* Use `r` to reset pan and zoom.
* Use `s` to take a screenshot.
* Use `f` to toggle fullscreen.
* Use `i` to toggle info text.
* Use `v` to toggle vsync.

Take a look at the [`config.txt`](fractals/assets/config.txt) file to see all available input keys, various fractal 
settings and how to specify custom colormap lists.


## Next Steps

- [x] Interactive visualization of the Mandelbrot Set fractal.
- [ ] Create an executable file via PyInstaller.
- [ ] Add Julia Set fractal rendering.


## Links

Mandelbrot Set - [Wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set)  
Plotting algorithms for the Mandelbrot set - [Wikipedia](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set)  
Collection of colormaps by Pratiman Patel - [GitHub.io](https://pratiman-91.github.io/colormaps)  
Collection of font files in .ttf format - [FontSquirrel](https://www.fontsquirrel.com)  

