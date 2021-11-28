# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 00:03:07 2021
"""

import logging
from functools import partial
from time import perf_counter
from typing import Optional

import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.diff import diff as cp_diff
from cvxpy.atoms.affine.reshape import reshape as cp_reshape
from cvxpy.atoms.norm1 import norm1 as cp_norm1
from scipy.fftpack import dct, idct
from tqdm import trange

from haar_transform import (haar_transform, haar_transform_2d,
                            inverse_haar_transform_2d)
from measurement_model import GIProcessingMethod
from misc import try_solving_until_success


logger = logging.getLogger("Fiber-GI-reduction.reduction")


class GIDenseReduction(GIProcessingMethod):
    """
    Measurement reduction method without additional information about the object.

    Parameters
    ----------
    model : GIMeasurementModel
        The ghost image measurement model on which to base the processing.

    Class attributes
    ----------------
    name : str
        Short name, typically used to refer to method's results when saving
        it to a file.
    desc : str
        Description of a method to use for plotting.
    """
    name = "red"
    desc = "Редукция измерений без дополнительной информации об объекте"

    def __call__(self, measurement, calc_cov_op: bool=False, eig: bool=False, # pylint: disable=W0221
                 **kwargs) -> np.ndarray:
        """
        Estimate the image using measurement reduction method without
        additional information.

        Parameters
        ----------
        measurement : array_like
            The measurement. If the measurement size is less than
            the available number of patterns, only the first
            `measurement.size` ones are used.
        calc_cov_op : bool, optional
            Whether to calculate the estimate covariance operator.
            The default is False.
        eig : bool, optional
            Whether to calculate the measurement model eigenbasis.
            If True, the value of calc_cov_op is ignored.
            The default is False.

        Returns
        -------
        numpy.ndarray
            The estimated image.

        """
        t_start = perf_counter()
        mt_op = self._mt_op(measurement.size)
        #TODO Change to iterative methods depending on which is faster
        u, s, vh = np.linalg.svd(mt_op, False, True, False)
        rcond = np.finfo(s.dtype).eps * max(u.shape[0], vh.shape[1])
        tol = np.amax(s) * rcond
        mask = abs(s) > tol
        s2 = np.zeros_like(s)
        s2[mask] = 1/s[mask]
        red_res = ((vh.T * s2) @ u.T @ measurement).reshape(
            self._measurement_model.img_shape
        )
        t_end = perf_counter()
        logger.info("Dense measurement reduction took %.3g s",
                    t_end - t_start)
        if eig: # pylint: disable=R1705
            return red_res, s2, vh
        elif calc_cov_op:
            return red_res, (vh.T * s2**2) @ vh
        else:
            return red_res


class GIDenseReductionIter(GIProcessingMethod):
    """
    Measurement reduction method without additional information about the object
    using Kaczmarz's iterative method.

    Parameters
    ----------
    model : GIMeasurementModel
        The ghost image measurement model on which to base the processing.

    Class attributes
    ----------------
    name : str
        Short name, typically used to refer to method's results when saving
        it to a file.
    desc : str
        Description of a method to use for plotting.
    """
    name = "rediter"
    desc = "Редукция измерений без дополнительной информации об объекте"

    def __call__(self, measurement, n_iter: Optional[int]=None, relax=0.15, # pylint: disable=W0221
                 start_from=None, print_progress=False, nonzero=False,
                 **kwargs) -> np.ndarray:
        """
        Estimate the image using measurement reduction method without
        additional information using Kaczmarz's iterative method.

        Parameters
        ----------
        measurement : array_like
            The measurement. If the measurement size is less than
            the available number of patterns, only the first
            `measurement.size` ones are used.
        n_iter : int or None, optional
            The number of iterations. If omitted, the number of iterations
            is measurement size.
        relax : float, optional
            The step size. Has to be below 2. The default is 0.15.
            Higher values improve convergence speed,
            but may result in overshooting or numerical instability.
        start_from : array_like or None, optional
            The initial approximation. If None, zero vector is used.

        Returns
        -------
        numpy.ndarray
            The estimated image.

        """
        mt_op = self._mt_op(measurement.size)
        if start_from is None:
            red_res = np.zeros(mt_op.shape[1])
        else:
            red_res = np.copy(start_from).ravel()
        if n_iter is None:
            n_iter = measurement.size
        for i in trange(n_iter):
            ind = i % mt_op.shape[0]
            row = mt_op[ind, :]
            correction_scal = (measurement[ind] - row.dot(red_res))/(row.dot(row))
            red_res += row * relax * correction_scal
            if nonzero:
                red_res = red_res.clip(0, None)
            # if print_progress:
                # print(i, correction_scal*row.dot(row))
            if isinstance(print_progress, list):
                print_progress.append(correction_scal*row.dot(row))
        return red_res.reshape(self._measurement_model.img_shape)


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
    logger.info("{}-related stuff took {:.3g} s".format(basis, t_transform_end - t_transform_start))
    return thresholded_data


class GISparseReduction(GIDenseReduction):
    """
    Measurement reduction method using the information about sparsity
    of the feature of interest of the studied object in a given basis.

    Parameters
    ----------
    model : GIMeasurementModel
        The ghost image measurement model on which to base the processing.

    Class attributes
    ----------------
    name : str
        Short name, typically used to refer to method's results when saving
        it to a file.
    desc : str
        Description of a method to use for plotting.
    """
    name = "reds"
    desc = "Редукция измерений при дополнительной информации об объекте"

    def __call__(self, measurement, thresholding_coeff: float=1.,
                 basis: str="eig", skip_tv: bool=False, **kwargs) -> np.ndarray:
        """
        Estimate the image using measurement reduction method using
        the information about sparsity of the feature of interest
        of the studied object in a given basis.

        Parameters
        ----------
        measurement : array_like
            The measurement. If the measurement size is less than
            the available number of patterns, only the first
            `measurement.size` ones are used.
        thresholding_coeff : float, optional
            The thresholding coefficient used for hypothesis testing.
            The default is 1. 0 corresponds to no thresholding.
        basis : {"eig", "dct", "haar"}, optional
            The basis of sparse representation. The default is "eig",
            corresponding to eigenbasis of the measurement model.
        skip_tv : bool, optional
            If True, skip the optimization of image total variation
            for estimating the null space component of the image.
            The default is False (do not skip).

        Returns
        -------
        numpy.ndarray
            The estimated image.

        """
        t_start = perf_counter()
        if basis == "eig":
            red_res, sing_val, sing_vec = super().__call__(measurement, eig=True)
            if thresholding_coeff > 0:
                red_res = do_thresholding(red_res, basis=basis,
                                          thresholding_coeff=thresholding_coeff,
                                        sing_val=sing_val, sing_vec=sing_vec)
        else:
            red_res, cov_op = super().__call__(measurement, calc_cov_op=True)
            if thresholding_coeff > 0:
                red_res = do_thresholding(red_res, cov_op, basis=basis,
                                          thresholding_coeff=thresholding_coeff)

        if skip_tv:
            return red_res
        #TODO Omit the following if sufficient measurement data to estimate the image
        mt_op = self._mt_op(measurement.size)
        expected_measurement = mt_op.dot(red_res.ravel())
        f = cp.Variable(mt_op.shape[1])
        f2 = cp_reshape(f, self._measurement_model.img_shape)
        sparsity_term = (cp_norm1(cp_diff(f2, k=1, axis=0))
                          + cp_norm1(cp_diff(f2, k=1, axis=1)))**2
        # sparsity_term = cp.atoms.total_variation.tv(f2)**2
        objective = cp.Minimize(sparsity_term)
        constraints = [mt_op @ f == expected_measurement]
        prob = cp.Problem(objective, constraints)
        #TODO Is there an easier way of trying all available solvers until one succeeds?
        try_solving_until_success(prob, [cp.OSQP, cp.ECOS, cp.SCS])
        t_end = perf_counter()
        logger.info("Sparse reduction took %.3g s for A shape %s",
                    t_end - t_start, mt_op.shape)
        return f.value.reshape(self._measurement_model.img_shape)
