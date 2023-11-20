import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Import custom modules
import fractals.color as fract_color
import fractals.mandelbrot as fract_mandelbrot

# Define default output directory
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), 'Pictures', 'FractalRenderingDebug')


# TODO : Clean up this file.
# TODO : Add Julia set fractals
# TODO : Combine histogram recoloring with smooth iterations count


# Define main function
def main():

	# Fractal settings
	img_size = (1000, 1000)
	max_iter = 64
	# Default view
	range_x = (-2, 1)
	range_y = (-1.5, 1.5)

	# Compute number of fractals iterations
	iterations = fract_mandelbrot.IterationsMandelbrotSet(img_size, range_x, range_y, max_iter)

	# PLOT - Show iterations count
	extent = np.hstack((range_x, range_y))
	plt.imshow(np.flipud(iterations), extent=extent, origin='upper', interpolation='none', cmap='viridis')
	plt.title('Mandelbrot set iterations')
	plt.colorbar()
	plt.show()


	# Compute image histograms
	hist = fract_color.IterationsHistogram(iterations, max_iter)
	hist_sum = np.sum(hist)

	# PLOT - Show both histograms
	plt.plot(hist / hist_sum, 'b-', label=f'Image size = {img_size[0]}x{img_size[1]}')
	plt.title('Histogram of fractals iterations')
	plt.legend()
	plt.show()

	# Histogram coloring
	iterations_norm = fract_color.HistogramRecoloring(iterations, hist, hist_sum)

	# Chose between original and recolored image
	iter = iterations / max_iter
	# iter = iterations_norm

	# Get colormap
	cmap = fract_color.GetColormapArray('balance')[:, 0:3]

	# Apply colormaps
	# img_color = fract_color.ApplyColormap_nearest(iter, cmap)
	img_color = fract_color.ApplyColormap_linear(iter, cmap)
	# img_color = fract_color.ApplyColormap_cubic(iter, cmap)

	# Convert to floats to RGB
	img_color = (255 * img_color).astype('uint8')

	# PLOT - Show the image
	extent = np.hstack((range_x, range_y))
	plt.imshow(np.flipud(img_color), extent=extent, origin='upper', interpolation='none')
	plt.show()

	# Save the image to a folder
	add_text = True
	pil_image = Image.fromarray(np.flipud(img_color))
	draw = ImageDraw.Draw(pil_image)
	if add_text:
		font_size = int(img_size[1] / 30)
		fill_color = (255, 255, 255)
		font = ImageFont.truetype('fractals/assets/NotoMono-Regular.ttf', font_size)
		draw.text((0, 0            ), f'BOUNDS-X : {range_x}', font=font, fill=fill_color)
		draw.text((0, font_size    ), f'BOUNDS-Y : {range_y}', font=font, fill=fill_color)
		draw.text((0, font_size * 2), f'NUM ITER : {max_iter}', font=font, fill=fill_color)
	filename = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
	os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
	pil_image.save(os.path.join(DEFAULT_OUTPUT_DIR, filename))


# Run main function
if __name__ == "__main__":
	main()
