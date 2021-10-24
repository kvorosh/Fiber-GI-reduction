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


def dense_reduction(measurement, mt_op, img_shape, calc_cov_op=False, eig=False):
    if calc_cov_op:
        print("Covariance operator calculation started.")
        t_start = perf_counter()
        #TODO Change to iterative methods depending on which is faster
        u, s, vh = np.linalg.svd(mt_op, False, True, False)
        mask = abs(s) > abs(s).max() * 1e-15
        s2 = np.zeros_like(s)
        s2[mask] = 1/s[mask]
        red_res = ((vh.T * s2) @ u.T @ measurement).reshape(img_shape)
        t_end = perf_counter()
        print("Cov. op. calculation took {:.3g} s".format(t_end - t_start))
        if eig:
            return red_res, s2, vh
        else:
            cov_op = (vh.T * s2**2) @ vh
            return red_res, cov_op
    else:
        red_res = lsmr(mt_op, measurement)[0].reshape(img_shape)
        return red_res


def dense_reduction_iter(measurement, mt_op, img_shape, n_iter=None, relax=0.15,
                         print_progress=False):
    red_res = np.zeros(mt_op.shape[1])
    if n_iter is None:
        n_iter = measurement.size

    for i in trange(n_iter):
        ind = i % mt_op.shape[0]
        row = mt_op[ind, :]
        correction_scal = (measurement[ind] - row.dot(red_res))/(row.dot(row))
        red_res += row * relax * correction_scal
        # red_res = red_res.clip(0, None)
        # if print_progress:
            # print(i, correction_scal*row.dot(row))
        if isinstance(print_progress, list):
            print_progress.append(correction_scal*row.dot(row))
    return red_res.reshape(img_shape)


def do_thresholding(data, cov_op=None, basis="haar", thresholding_coeff=0., kind="hard",
                    sing_val=None, sing_vec=None):
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
    cov_op : (N, N) array_like or None
        Covariance operator of input data.
        Can be None if basis == "eig".
    basis : {"haar", "dct", "eig"}
        Thresholding basis choice.
    threshold_coeff : float
        Multiplier for variances.
    kind : {"hard", "soft"}
        Whether to do hard or soft thresholding.
    sing_val : array_like or None
        Singular values of the measuring system.
        Used only if basis == "eig".
    sing_vec : array_like or None
        Singular vectors of the measuring system.
        Used only if basis == "eig".

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
    elif basis == "eig":
        transform_2d = lambda data: sing_vec.dot(data.ravel())
        inverse_transform_2d = lambda matr: (sing_vec.T.dot(matr)).reshape(data.shape)
    else:
        raise NotImplementedError("Thresholding in {} basis not implemented yet!".format(basis))

    # The following should work when data is both 1D and 2D.
    if basis == "eig":
        variances = sing_val**2
    else:
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



def sparse_reduction(measurement, mt_op, img_shape, thresholding_coeff=1.,
                     basis="eig"):
    #TODO Skip thresholding if no noise
    if basis == "eig":
        red_res, sing_val, sing_vec = dense_reduction(
            measurement, mt_op, img_shape, calc_cov_op=True, eig=True
        )
        if thresholding_coeff == 0:
            return red_res
        print("Dense reduction done")
        res = do_thresholding(red_res, basis=basis,
                              thresholding_coeff=thresholding_coeff,
                              sing_val=sing_val, sing_vec=sing_vec)
    else:
        red_res, cov_op = dense_reduction(measurement, mt_op, img_shape,
                                                  calc_cov_op=True)
        print("Dense reduction done")
        res = do_thresholding(red_res, cov_op, basis=basis,
                              thresholding_coeff=thresholding_coeff)

    #TODO Omit the following if sufficient measurement data to estimate the image
    expected_measurement = mt_op.dot(res.ravel())
    f = cp.Variable(mt_op.shape[1])
    f2 = cp_reshape(f, img_shape)
    sparsity_term = (cp_norm1(cp_diff(f2, k=1, axis=0))
                      + cp_norm1(cp_diff(f2, k=1, axis=1)))**2
    # sparsity_term = cp.atoms.total_variation.tv(f2)**2
    objective = cp.Minimize(sparsity_term)
    constraints = [mt_op @ f == expected_measurement]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS)
    except cp.error.SolverError:
        prob.solve(solver=cp.ECOS)
    t_end = perf_counter()
    return f.value.reshape(img_shape)
