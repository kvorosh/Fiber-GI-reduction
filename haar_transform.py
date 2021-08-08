# -*- coding: utf-8 -*-
"""
Haar transform and inverse Haar transform.
"""


import numpy as np


def haar_transform(x, axis=0, inplace=False):  # pylint: disable=C0103
    """
    1D Haar transform.

    This transform version is orthogonal, but not orthonormal.
    To obtain the matrix describing it, pass np.eye of suitable size as `x`.

    Parameters
    ----------
    x : ndarray
        Data to be transformed. Its size along the specified axis must be
        a power of 2.
    axis : int
        The axis along which transform is applied.
    inplace : bool
        Whether to do the transform in-place, reducing memory requirements
        but usually overwriting the original data.

    Returns
    -------
    haar : ndarray
        Transform result with the same shape as `x`.
    """
    x = np.swapaxes(x, 0, axis)
    if not inplace:
        x = x.copy()

    n = len(x)  # pylint: disable=C0103

    tmp = np.empty_like(x)
    avg = tmp[: n//2, ...]
    dif = tmp[n//2:, ...]

    while n > 1:
        avg = (x[::2, ...] + x[1::2, ...])/2
        dif = x[::2, ...] - avg

        x[: n//2, ...] = avg[: n//2, ...]
        x[n//2: n, ...] = dif[: n//2, ...]

        n //= 2  # pylint: disable=C0103

    x = np.swapaxes(x, 0, axis)
    return x


def inverse_haar_transform(x, axis=0, inplace=False):  # pylint: disable=C0103
    """
    1D inverse Haar transform.

    This transform version is orthogonal, but not orthonormal.
    To obtain the matrix describing it, pass np.eye of suitable size as `x`.

    Parameters
    ----------
    x : ndarray
        Data to be transformed. Its size along the specified axis must be
        a power of 2.
    axis : int
        The axis along which transform is applied.
    inplace : bool
        Whether to do the transform in-place, reducing memory requirements
        but usually overwriting the original data.

    Returns
    -------
    unhaar : ndarray
        Transform result with the same shape as `x`.
    """
    x = np.swapaxes(x, 0, axis)
    if not inplace:
        x = x.copy()

    n = len(x)  # pylint: disable=C0103
    tmp = np.zeros_like(x)

    count = 2
    while count <= n:
        tmp[: count: 2, ...] = x[: count//2, ...] + x[count//2: count, ...]
        tmp[1: count: 2, ...] = x[: count//2, ...] - x[count//2: count, ...]
        x[: count, ...] = tmp[: count, ...]
        count *= 2
    return np.swapaxes(tmp, 0, axis)


def haar_transform_2d(data, inplace=False):
    """
    2D Haar transform along first two axes.

    This transform version is orthogonal, but not orthonormal.

    Parameters
    ----------
    data : ndarray
        Input data. First two axes sizes must be powers of 2.
    inplace : bool
        Whether to do the transform in-place, reducing memory requirements
        but usually overwriting the original data.

    Returns
    -------
    haar : ndarray
        Transform result with the same shape as `data`.

    """
    haar = haar_transform(data, 0, inplace)
    haar = haar_transform(haar, 1, True)
    return haar


def inverse_haar_transform_2d(data, inplace=False):
    """
    2D inverse Haar transform along first two axes.

    This transform version is orthogonal, but not orthonormal.

    Parameters
    ----------
    data : ndarray
        Input data. First two axes sizes must be powers of 2.
    inplace : bool
        Whether to do the transform in-place, reducing memory requirements
        but usually overwriting the original data.

    Returns
    -------
    unhaar : ndarray
        Transform result with the same shape as `data`.

    """
    unhaar = inverse_haar_transform(data, 0, inplace)
    unhaar = inverse_haar_transform(unhaar, 1, True)
    return unhaar


