import numba
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader

def read_shader_source(path_to_shader):
    with open(path_to_shader, 'r') as f:
        shader_source = f.read()
    return shader_source

def create_shader_program(vertex_src, fragment_src):
    # Compile the shaders
    vertex_shader = compileShader(vertex_src, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_src, GL_FRAGMENT_SHADER)
    # Create a program and link the shaders
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    # Delete the shaders and return the program
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def get_uniform_locations(shader_program, uniform_names):
    uniform_locations = {}
    for uniform_name in uniform_names:
        uniform_locations[uniform_name] = glGetUniformLocation(shader_program, uniform_name)
    return uniform_locations

@numba.njit(cache=True)
def texture_transform_mat(pos_x, pos_y, size_x, size_y):
    trans_mat = np.zeros(shape=(4, 4), dtype='float')
    trans_mat[0, 0] = size_x
    trans_mat[1, 1] = size_y
    trans_mat[3, 0] = pos_x
    trans_mat[3, 1] = pos_y
    trans_mat[3, 3] = 1.0
    return trans_mat

@numba.njit(cache=True)
def ortho_transform_mat(left, right, bottom, top, near, far):
    trans_mat = np.zeros(shape=(4, 4), dtype='float')
    trans_mat[0, 0] = 2.0 / (right - left)
    trans_mat[1, 1] = 2.0 / (top - bottom)
    trans_mat[2, 2] = -2.0 / (far - near)
    trans_mat[3, 0] = -(right + left) / (right - left)
    trans_mat[3, 1] = -(top + bottom) / (top - bottom)
    trans_mat[3, 2] = -(far + near) / (far - near)
    trans_mat[3, 3] = 1.0
    return trans_mat
