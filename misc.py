# -*- coding: utf-8 -*-
"""
@author: balakin
"""

import numpy as np
from scipy.linalg import toeplitz
from imageio import imread, imwrite
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import norm


def load_demo_image(img_id=0, full_span=False, pad_by=0):
    """
    Load the image that is to be used for demonstrations.

    Parameters
    ----------
    img_id : int
        ID of the image to load
    full_span : bool
        If True, linearly 'stretch' pixel brightnesses so that they span
        the entire [0, 1] interval.
    pad_by : int
        Pad the image by `pad_by` in all directions.

    Returns
    -------
    img : ndarray
        The loaded grayscale image.
    """
    img_paths = {0: "teapot.png", 1: "logo.png", 2: "logo64.png",
                 3: partial(demo_two_slits, (64, 64), 48, 8, 8), 4: "simple.png",
                 5: "alum.png", 6: "teapot128.png", 7: "teapot64.png",
                 8: partial(demo_two_slits, (128, 128), 96, 16, 16)}
    if callable(img_paths[img_id]):
        img = img_paths[img_id]()
    else:
        img = imread(img_paths[img_id], as_gray=True)
        img -= 0.4999
        img /= 255
        if full_span:
            img -= img.min()
            img /= img.max()
    if pad_by:
        old_img = img
        img = np.zeros((img.shape[0] + 2*pad_by, img.shape[1] + 2*pad_by))
        img[pad_by: pad_by + old_img.shape[0], pad_by: pad_by + old_img.shape[1]] = old_img
    return img


def demo_two_slits(img_shape, slit_length, slit_distance, slit_width,
                   vertical=True):
    image = np.zeros(img_shape, dtype=bool)
    if vertical:
        slice_axis_0 = slice((img_shape[0] - slit_length)//2, (img_shape[0] + slit_length)//2)
        slice_axis_1a = slice((img_shape[1] - slit_distance - 2*slit_width)//2, (img_shape[1] - slit_distance)//2)
        slice_axis_1b = slice((img_shape[1] + slit_distance)//2, (img_shape[1] + slit_distance + 2*slit_width)//2)
        image[slice_axis_0, slice_axis_1a] = 1
        image[slice_axis_0, slice_axis_1b] = 1
    else:
        raise NotImplementedError("horizontal slits not implemented yet!")
    return image


def save_image_for_show(img_data, img_filename: str, unit_scale=False, rescale=False):
    """
    Save an image to a file with specified name. Output format is PNG.

    Parameters
    ----------
    img_data : ndarray
        An image
    img_filename : str
        The name of output file without extension.
    unit_scale : bool
        Whether the pixels take values in [0, 1] range.
    rescale : bool
        Whether to move the pixel values to the [0, 255] range.
    """
    if rescale:
        img_data = img_data - img_data.min()
        img_data *= 255 / img_data.max()
        img_data = img_data.astype(np.uint8)
    if unit_scale:
        img_data = (img_data * 255).astype(np.uint8)
    imwrite("../figures/" + img_filename + ".png", img_data)


def visualize_image(image, vmax=None):
    """
    Visualize a grayscale image.

    Parameters
    ----------
    image : (N, M) ndarray
        The image
    vmax : {int, float, None}
        The maximal pixel brightness. If None, image maximum is used.
    """
    plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=vmax)


def save_vectors(fname, vectors, names, with_ind=True, num_fmt="%.18e"):
    """
    Save several vectors of the same length to file with headers.
    """
    if with_ind:
        fmt = ["%d"] + [num_fmt]*len(vectors)
        ind = np.arange(vectors[0].size) + 1
        vectors = [ind] + vectors
        names = ["ind"] + names
    else:
        fmt = num_fmt

    results = np.vstack(vectors).T
    header = "\t".join(names)
    np.savetxt("../data/" + fname + ".dat", results, delimiter='\t',
               header=header, fmt=fmt)
