# -*- coding: utf-8 -*-
"""
@author: balakin
"""

import logging
from functools import partial
from typing import Sequence

import cvxpy as cp
from scipy.sparse import coo_array
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite


logger = logging.getLogger("FGI-red.misc")


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


def apply_mask_to_mt_op(mt_op, mask):
    flat_mask = mask.ravel()
    mt_op_masked = np.zeros((mt_op.shape[0], np.count_nonzero(flat_mask) + 1))
    mt_op_masked[:, 0: -1] = mt_op[:, flat_mask]
    mt_op_masked[:, -1] = mt_op[:, ~flat_mask].sum(axis=1)
    return mt_op_masked


def transform_using_mask(mask):
    flat_mask = mask.ravel()
    nnz = np.count_nonzero(flat_mask)
    ind0 = np.hstack((np.where(flat_mask)[0], np.where(~flat_mask)[0]))
    ind1 = np.hstack((np.arange(nnz), nnz*np.ones(flat_mask.size - nnz, dtype=ind0.dtype)))
    values = np.ones(ind0.size, dtype=bool)
    transform = coo_array((values, (ind0, ind1)))
    return transform


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


def try_solving_until_success(problem: cp.problems.problem.Problem,
                              solvers: Sequence[str], check_value=None, **kwargs):
    """
    Try using different solvers on the optimization problem until one succeeds.
    The first used one is the default one, followed by those listed in `solvers`
    in that order. The default solver is not included twice.

    Parameters
    ----------
    problem : cvxpy.problems.problem.Problem
        The optimization problem.
    solvers : sequence of str
        The solvers.
    **kwargs
        Arguments to be passed to problem.solve()

    Returns
    -------
    None.

    """
    try:
        problem.solve(**kwargs)
        if check_value is not None and check_value.value is None:
            logger.info("Result is None for the default method")
            raise cp.error.SolverError
    except cp.error.SolverError as e:
        try:
            solvers.remove(problem.solver_stats.solver_name)
        except ValueError:
            pass
        except AttributeError:
            solvers = [s for s in solvers if s not in str(e)]
        for solver in solvers:
            try:
                problem.solve(solver=solver, **kwargs)
                if check_value is not None and check_value.value is None:
                    logger.info(f"Result is None for method {solver}")
                    raise cp.error.SolverError
            except cp.error.SolverError:
                pass
            else:
                break
