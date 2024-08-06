import numpy as np
from skimage import color

def colorspace(conversion, *args):
    if len(args) < 1:
        raise ValueError('Not enough input arguments.')
    
    src_space, dest_space = parse_conversion(conversion)

    if len(args) == 1:
        image = args[0]
    elif len(args) >= 3:
        image = np.stack(args, axis=-1)
    else:
        raise ValueError('Invalid number of input arguments.')
    
    flip_dims = (image.ndim == 2)
    if flip_dims:
        image = np.expand_dims(image, axis=-1)
    if image.dtype != np.float64:
        image = image.astype(np.float64) / 255.0
    
    if src_space == 'rgb' and dest_space == 'hsv':
        result = color.rgb2hsv(image)
    elif src_space == 'hsv' and dest_space == 'rgb':
        result = color.hsv2rgb(image)
    elif src_space == 'rgb' and dest_space == 'lab':
        result = color.rgb2lab(image)
    elif src_space == 'lab' and dest_space == 'rgb':
        result = color.lab2rgb(image)
    elif src_space == 'rgb' and dest_space == 'hsl':
        result = rgb_to_hsl(image)
    elif src_space == 'hsl' and dest_space == 'rgb':
        result = hsl_to_rgb(image)
    else:
        raise ValueError(f'Conversion from {src_space} to {dest_space} not implemented.')
    
    if flip_dims:
        result = np.squeeze(result, axis=-1)
    if len(args) > 1:
        return np.split(result, 3, axis=-1)
    else:
        return result

def parse_conversion(conversion):
    if '->' in conversion:
        src_space, dest_space = conversion.split('->')
    elif '<-' in conversion:
        dest_space, src_space = conversion.split('<-')
    else:
        raise ValueError(f"Invalid conversion format '{conversion}'.")
    return src_space.strip().lower(), dest_space.strip().lower()


def rgb_to_hsl(image):
    image = color.rgb2hsv(image)
    h = image[:, :, 0]
    s = image[:, :, 1]
    v = image[:, :, 2]
    l = (2 - s) * v / 2
    s = np.where(l < 0.5, s * v / (l * 2), s * v / (2 - l * 2))
    return np.stack([h, s, l], axis=-1)

def hsl_to_rgb(image):
    h = image[:, :, 0]
    s = image[:, :, 1]
    l = image[:, :, 2]
    v = np.where(l < 0.5, l * (1 + s), (l + s - l * s))
    s = 2 * (v - l) / v
    rgb = color.hsv2rgb(np.stack([h, s, v], axis=-1))
    return rgb
