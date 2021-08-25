# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 00:03:07 2021
"""

from functools import partial
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import lsmr
from scipy.fftpack import dct, idct
from haar_transform import haar_transform, haar_transform_2d, inverse_haar_transform_2d
import cvxpy as cp
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.atoms.affine.diff import diff as cp_diff
from cvxpy.atoms.affine.reshape import reshape as cp_reshape
from cvxpy.atoms import norm as cp_norm
from cvxpy.atoms.norm1 import norm1 as cp_norm1
from tqdm import trange


def dense_reduction(measurement, mt_op, img_shape, calc_cov_op=False):
    red_res = lsmr(mt_op, measurement)[0].reshape(img_shape)
    if calc_cov_op:
        print("Covariance operator calculation started.")
        t_start = perf_counter()
        cov_op = np.linalg.pinv(mt_op.T.dot(mt_op))
        t_end = perf_counter()
        print("Cov. op. calculation took {:.3g} s".format(t_end - t_start))
        return red_res, cov_op
    else:
        return red_res


def do_thresholding(data, cov_op, basis="haar", thresholding_coeff=0., kind="hard"):
    """
    Apply thresholding to data based on its variance.

    Data values whose absolute value in given basis
    is below their variance times `threshold_coeff` are pruned.
    Furthermore, in case of soft thresholding remaining values
    are shifted towards zero by thresholding values.

    Parameters
    ----------
    data : array_like
        Input data. Its size must be equal to N (see below).
    cov_op : (N, N) array_like
        Covariance operator of input data.
    basis : {"haar", "dct"}
        Thresholding basis choice.
    threshold_coeff : float
        Multiplier for variances.
    kind : {"hard", "soft"}
        Whether to do hard or soft thresholding.

    Returns
    -------
    thresholded_data : ndarray
        Data after thresholding.
    """
    t_transform_start = perf_counter()
    if basis == "haar":
        transform_1d = partial(haar_transform, axis=0)
        transform_2d = haar_transform_2d
        inverse_transform_2d = inverse_haar_transform_2d
    elif basis == "dct":
        transform_1d = partial(dct, axis=0)
        transform_2d = lambda data: dct(dct(data, axis=0), axis=1)
        inverse_transform_2d = lambda data: idct(idct(data, axis=1)/4, axis=0)/data.size
    else:
        raise NotImplementedError("Thresholding in {} basis not implemented yet!".format(basis))

    # The following should work when data is both 1D and 2D.
    transform_matrix = transform_1d(np.eye(data.shape[0]))
    if len(data.shape) > 1:
        if data.shape[0] != data.shape[1]:
            transform_matrix_1 = transform_1d(np.eye(data.shape[1]))
        else:
            transform_matrix_1 = transform_matrix
        transform_matrix = np.kron(transform_matrix, transform_matrix_1)
    variances = transform_matrix.dot(cov_op).dot(transform_matrix.T).diagonal().reshape(data.shape)
    data_in_basis = transform_2d(data)

    threshold = np.sqrt(variances)*thresholding_coeff
    mask = abs(data_in_basis) < threshold

    thresholded_data_in_basis = data_in_basis.copy()
    thresholded_data_in_basis[mask] = 0

    if kind == "soft":
        mask_pos = np.logical_and(~mask, thresholded_data_in_basis > 0)
        thresholded_data_in_basis[mask_pos] -= threshold[mask_pos]
        mask_neg = np.logical_and(~mask, thresholded_data_in_basis < 0)
        thresholded_data_in_basis[mask_neg] += threshold[mask_neg]

    thresholded_data = inverse_transform_2d(thresholded_data_in_basis)
    t_transform_end = perf_counter()
    print("{}-related stuff took {:.3g} s".format(basis, t_transform_end - t_transform_start))
    return thresholded_data



def sparse_reduction(measurement, mt_op, img_shape, thresholding_coeff=1., data_marker=None):
    if data_marker is None:
        red_res, cov_op = dense_reduction(measurement, mt_op, img_shape, calc_cov_op=True)
    else:
        try:
            with np.load("res_cov_{}.npz".format(data_marker)) as data:
                red_res = data["red"]
                cov_op = data["cov"]
        except FileNotFoundError:
            red_res, cov_op = dense_reduction(measurement, mt_op, img_shape, calc_cov_op=True)
            np.savez_compressed("res_cov_{}.npz".format(data_marker), red=red_res, cov=cov_op)
    print("Dense reduction done")
    res = do_thresholding(red_res, cov_op, basis="dct",
                          thresholding_coeff=thresholding_coeff)
    return res
