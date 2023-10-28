import numba
import numpy as np

'''
O-------------------------------------------------------------------------------O
| COLOR SPACE TRANSFORMATIONS                                                   |
O-------------------------------------------------------------------------------O
'''

@numba.njit(cache=True)
def HSV2RGB(hsv):
    h, s, v = hsv
    c = v * s
    h_div_60 = h / 60.0
    x = c * (1.0 - abs(h_div_60 % 2 - 1))
    m = v - c
    if (0.0 <= h_div_60 and h_div_60 < 1.0):
        R1, G1, B1 = c, x, 0.0
    elif (1.0 <= h_div_60 and h_div_60 < 2.0):
        R1, G1, B1 = x, c, 0.0
    elif (2.0 <= h_div_60 and h_div_60 < 3.0):
        R1, G1, B1 = 0, c, x
    elif (3.0 <= h_div_60 and h_div_60 < 4.0):
        R1, G1, B1 = 0, x, c
    elif (4.0 <= h_div_60 and h_div_60 < 5.0):
        R1, G1, B1 = x, 0.0, c
    else:
        R1, G1, B1 = c, 0.0, x
    return R1 + m, G1 + m, B1 + m


@numba.njit(cache=True)
def HSV2RGB_alt(hsv):
    h, s, v = hsv
    c = v * s
    h_div_60 = h / 60.0
    k1 = (1.0 + h_div_60) % 6
    k3 = (3.0 + h_div_60) % 6
    k5 = (5.0 + h_div_60) % 6
    f1 = v - c * max(0.0, min(k1, min(4.0 - k1, 1.0)))
    f3 = v - c * max(0.0, min(k3, min(4.0 - k3, 1.0)))
    f5 = v - c * max(0.0, min(k5, min(4.0 - k5, 1.0)))
    return f5, f3, f1


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
| APPLY COLORMAPS                                                               |
O-------------------------------------------------------------------------------O
'''

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
            temp_color = interpolate_linear(t, cmap[n0, :], cmap[n1, :])
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
            img_color[i, j, :] = interpolate_cubic(t, cmap[n0, :], cmap[n1, :], cmap[n2, :], cmap[n3, :])
    return img_color


'''
O-------------------------------------------------------------------------------O
| INTERPOLATION FUNCTIONS                                                       |
O-------------------------------------------------------------------------------O
'''

# Linear interpolation of 2 points
@numba.njit(cache=True)
def interpolate_linear(t, v0, v1):
    return (1.0 - t) * v0 + t * v1


# Cubic interpolation of 4 points
@numba.njit(cache=True)
def interpolate_cubic(t, v0, v1, v2, v4):
    return v1 + 0.5 * t * (v2 - v0 + t * (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v4 + t * (3.0 * (v1 - v2) + v4 - v0)))


'''
O-------------------------------------------------------------------------------O
| CUSTOM COLORMAPS                                                              |
O-------------------------------------------------------------------------------O
'''

def cmap_wikipedia():
    cmap = np.asarray([[ 66,  30,  15, 255],
                       [ 25,   7,  26, 255],
                       [  9,   1,  47, 255],
                       [  4,   4,  73, 255],
                       [  0,   7, 100, 255],
                       [ 12,  44, 138, 255],
                       [ 24,  82, 177, 255],
                       [ 57, 125, 209, 255],
                       [134, 181, 229, 255],
                       [211, 236, 248, 255],
                       [241, 233, 191, 255],
                       [248, 201,  95, 255],
                       [255, 170,   0, 255],
                       [204, 128,   0, 255],
                       [153,  87,   0, 255],
                       [106,  52,   3, 255]])
    return (cmap / 255).astype('float32')
