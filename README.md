# FractalRendering
Interactive app for visualization of the Mandelbrot Set written in Python and GLSL.  

![main_app.py](docs/main_app.gif)


## Features

* GPU-accelerated rendering via OpenGL 4.0.
* Double precision arithmetic with zoom up to 10^15.
* Interactive panning and zooming with the mouse.
* Arbitrary window resizing, full-screen and vsync support.
* Automatic DPI-scaling and variable render size.
* Fully customizable settings file.
* 296 available colormaps.


## How to Use?

The app is tested and works on Windows 11 and Linux Debian 11. The app should also work on MacOS systems that that still
support OpenGL. 

Please make sure you have installed the latest version of [Python](https://www.python.org/downloads/) before continuing. Furthermore, it is advisable
to use python and install modules inside a [virtual environment](https://docs.python.org/3/library/venv.html). Copy the following commands into the terminal:   

1. Clone the repository:  
   ```sh  
   git clone git@github.com:MarkoLeskovar/FractalRendering.git 
   cd FractalRendering/
   ```

2. Install required the python modules from [`requirements.txt`](requirements.txt):  
   ```sh  
   pip install -r requirements.txt
   ```
   
3. Run the main script to run the interactive app:
   ```sh  
   python main_app.py
   ```

If you are experiencing low FPS while running the app on a laptop with both integrated and dedicated GPUs, make sure to 
enable the dedicated GPU in your
[Nvidia Control Panel](https://www.nvidia.com/content/Control-Panel-Help/vLatest/en-us/mergedProjects/nv3d/Setting_the_Preferred_Graphics_Processor.htm) 
or [AMD Radeon](https://www.amd.com/en/support/kb/faq/dh2-024) software.


## Controls & Customization

Fully customizable list of all available app settings and controls is available in the [`config.json`](fractals/assets/config.json) file. Bellow
you will find some of the most used keys.

* Use `ARROW KEY UP` / `DOWN` or `MOUSE SCROLL` to zoom in and out.
* Use `ARROW KEY LEFT` / `RIGHT` to change colormaps.
* Use `+` / `-` to increase or decrease the number of fractal iterations.
* Use `*` / `/` to increase or decrease the pixel scaling.
* Use `R` to reset pan and zoom.
* Use `S` to take a screenshot.
* Use `F` to toggle fullscreen.
* Use `I` to toggle info text.
* Use `V` to toggle vsync.


## Next Steps

- [x] Interactive visualization of the Mandelbrot Set fractal.
- [ ] Create an executable file via PyInstaller.
- [ ] Add Julia Set fractal rendering.


## Links

Mandelbrot Set - [Wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set)  
Plotting algorithms for the Mandelbrot set - [Wikipedia](https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set)  
Collection of colormaps by Pratiman Patel - [GitHub.io](https://pratiman-91.github.io/colormaps)  
Collection of font files in .ttf format - [FontSquirrel](https://www.fontsquirrel.com)  

