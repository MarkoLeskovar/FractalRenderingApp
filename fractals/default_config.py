import os

__all__ = [
    'DEFAULT_APP_CONFIG',
    'DEFAULT_CONTROLS_CONFIG',
    'DEFAULT_FRACTAL_CONFIG',
    'DEFAULT_CMAPS',
    'DEFAULT_OUTPUT_DIR'
]

DEFAULT_APP_CONFIG = {
        'WIN_WIDTH':            800,
        'WIN_HEIGHT':           600,
        'PIX_SCALE_MIN':        0.5,
        'PIX_SCALE_MAX':        8.0,
        'PIX_SCALE_STEP':       0.25,
        'FONT_SIZE':            20,
        'FONT_FILE':            "NotoMono-Regular.ttf",
}

DEFAULT_CONTROLS_CONFIG = {
    'EXIT':                 'KEY_ESCAPE',
    'INFO':                 'KEY_I',
    'VSYNC':                'KEY_V',
    'FULLSCREEN':           'KEY_F',
    'SCREENSHOT':           'KEY_S',
    'ZOOM_IN':              'KEY_UP',
    'ZOOM_OUT':             'KEY_DOWN',
    'SHIFT_VIEW':           'MOUSE_BUTTON_LEFT',
    'RESET_VIEW':           'KEY_R',
    'ITER_INCREASE':        'KEY_KP_ADD',
    'ITER_DECREASE':        'KEY_KP_SUBTRACT',
    'PIX_SCALE_INCREASE':   'KEY_KP_MULTIPLY',
    'PIX_SCALE_DECREASE':   'KEY_KP_DIVIDE',
    'CMAP_NEXT':            'KEY_RIGHT',
    'CMAP_PREV':            'KEY_LEFT',
}

DEFAULT_FRACTAL_CONFIG = {
    'MANDELBROT': {
        'RANGE_X_MIN':     -2.0,
        'RANGE_X_MAX':      1.0,
        'NUM_ITER':         64,
        'NUM_ITER_MIN':     64,
        'NUM_ITER_MAX':     2048,
        'NUM_ITER_STEP':    32,
    }
}

DEFAULT_CMAPS = [
    'balance',
    'cet_l_tritanopic_krjcw1_r',
    'cet_l_protanopic_deuteranopic_kbw_r',
    'jungle_r',
    'apple_r'
]

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), 'Pictures', 'FractalRendering')
