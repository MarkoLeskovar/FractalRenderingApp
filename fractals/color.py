import numba
import numpy as np
import colormaps as cmaps


'''
O-------------------------------------------------------------------------------O
| HISTOGRAM RECOLORING                                                          |
O-------------------------------------------------------------------------------O
'''

@numba.njit(cache=True)
def IterationsHistogram(img_iterations, max_iter):
    # max_iter_max_id = max_iter - 1
    histogram = np.zeros(shape=max_iter, dtype='int32')
    for i in range(img_iterations.shape[0]):
        for j in range(img_iterations.shape[1]):
            # n = min(int(img_iterations[i, j]), max_iter_max_id)
            n = int(img_iterations[i, j]) - 1
            histogram[n] += 1
    return histogram


@numba.njit(cache=True)
def HistogramRecoloring(img_iterations, histogram, histogram_sum):
    img_iterations_norm = np.zeros(shape=img_iterations.shape, dtype='float')
    for i in range(img_iterations.shape[0]):
        for j in range(img_iterations.shape[1]):
            temp_iter = img_iterations[i, j]
            for n in range(int(temp_iter)):
                img_iterations_norm[i, j] += histogram[n] / histogram_sum
    return img_iterations_norm


'''
O-------------------------------------------------------------------------------O
| COLORMAPS                                                                     |
O-------------------------------------------------------------------------------O
'''

def LoadColormapsFile(cmap_file):
    # Open the file
    with open(cmap_file, mode='r') as f:
        lines_raw = f.readlines()
    # Parse the lines
    lines = []
    for line in lines_raw:
        line = line.split('#')[0]
        line = line.strip('\n').strip('\n').lstrip('\t').replace(' ', '')
        if len(line) != 0:
            lines.append(line)
    return lines


def GetColormapArray(cmap_name):
    cmap = getattr(cmaps, cmap_name)
    return cmap(range(256))


@numba.njit(cache=True)
def ApplyColormap_nearest(iterations, cmap):
    img_size = iterations.shape
    cmap_max_i = cmap.shape[0] - 1
    img_color = np.empty(shape=(img_size[0], img_size[1], 3), dtype='float')
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            index = int(np.round(iterations[i, j] * cmap_max_i))
            img_color[i, j, :] = cmap[index, :]
    return img_color


@numba.njit(cache=True)
def ApplyColormap_linear(iterations, cmap):
    img_size = iterations.shape
    cmap_max_i = cmap.shape[0] - 1
    img_color = np.empty(shape=(img_size[0], img_size[1], 3), dtype='float')
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            index = iterations[i, j] * cmap_max_i
            n0 = int(index)
            n1 = min(n0 + 1, cmap_max_i)
            t = index % 1
            temp_color = _interpolate_linear(t, cmap[n0, :], cmap[n1, :])
            img_color[i, j, :] = temp_color
    return img_color


@numba.njit(cache=True)
def ApplyColormap_cubic(iterations, cmap):
    img_size = iterations.shape
    cmap_max_i = cmap.shape[0] - 1
    img_color = np.empty(shape=(img_size[0], img_size[1], 3), dtype='float')
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            index = iterations[i, j] * cmap_max_i
            n1 = int(index)
            n0 = max(n1 - 1, 0)
            n2 = min(n1 + 1, cmap_max_i)
            n3 = min(n1 + 2, cmap_max_i)
            t = index % 1
            img_color[i, j, :] = _interpolate_cubic(t, cmap[n0, :], cmap[n1, :], cmap[n2, :], cmap[n3, :])
    return img_color


'''
O-------------------------------------------------------------------------------O
| PRIVATE - INTERPOLATION FUNCTIONS                                             |
O-------------------------------------------------------------------------------O
'''

# Linear interpolation of 2 points
@numba.njit(cache=True)
def _interpolate_linear(t, v0, v1):
    return (1.0 - t) * v0 + t * v1


# Cubic interpolation of 4 points
@numba.njit(cache=True)
def _interpolate_cubic(t, v0, v1, v2, v4):
    return v1 + 0.5 * t * (v2 - v0 + t * (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v4 + t * (3.0 * (v1 - v2) + v4 - v0)))
