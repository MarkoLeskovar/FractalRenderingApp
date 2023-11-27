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
