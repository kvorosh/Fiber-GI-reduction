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


logger = logging.getLogger("FGI-red.reduction")


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
                 downscale_factors=None, use_mask: Optional[bool]=False,
                 keep_1d: Optional[bool]=False,
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
        downscale_factors : 2-tuple of ints or None, optional
            If not None, downscale the produced image in the specified way.
        use_mask : bool, optional
            If True, consider the values of all pixels with weak transmittance
            to have the same brightness value.
        keep_1d : bool, optional
            If True, keep the 1D shape of the result.

        Returns
        -------
        numpy.ndarray
            The estimated image.

        """
        t_start = perf_counter()
        mt_op = self._mt_op(measurement.size, downscale_factors, use_mask)
        #TODO Change to iterative methods depending on which is faster
        u, s, vh = np.linalg.svd(mt_op, False, True, False)
        rcond = np.finfo(s.dtype).eps * max(u.shape[0], vh.shape[1])
        tol = np.amax(s) * rcond
        mask = abs(s) > tol
        s2 = np.zeros_like(s)
        s2[mask] = 1/s[mask]
        red_res = (vh.T * s2) @ u.T @ measurement
        if not keep_1d:
            red_res = self.to_image(red_res, downscale_factors, use_mask)
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
                 return_cond=None,
                 downscale_factors=None, use_mask: Optional[bool]=False,
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
        return_cond : optional
            Return the intermediate results obtained after ith iteration
            if return_cond(i) is True. If None, only the final result is returned.
        downscale_factors : 2-tuple of ints or None, optional
            If not None, downscale the produced image in the specified way.
        use_mask : bool, optional
            If True, consider the values of all pixels with weak transmittance
            to have the same brightness value.

        Returns
        -------
        result : numpy.ndarray
            The estimated image.
        intermediate_results : list of (int, numpy.ndarray)
            The intermediate results after the specified number of iterations.
        """
        mt_op = self._mt_op(measurement.size, downscale_factors, use_mask)
        if start_from is None:
            red_res = np.zeros(mt_op.shape[1])
        else:
            red_res = np.copy(start_from).ravel()
        if n_iter is None:
            n_iter = measurement.size
        if return_cond is not None:
            intermediate_results = []
        for i in range(n_iter):
            ind = i % mt_op.shape[0]
            row = mt_op[ind, :]
            correction_scal = (measurement[ind] - row.dot(red_res))/(row.dot(row))
            red_res += row * relax * correction_scal
            if nonzero:
                red_res = red_res.clip(0, None)
            if isinstance(print_progress, list):
                print_progress.append(correction_scal*row.dot(row))
            if return_cond is not None and return_cond(i):
                # If one does not copy, "+=" above results
                # in the same vector added to intermediate results for all iterations
                intermediate_results.append((
                    i, self.to_image(np.copy(red_res), downscale_factors, use_mask)
                ))
        result = self.to_image(red_res, downscale_factors, use_mask)
        if return_cond is None:
            return result
        else:
            return result, intermediate_results


def do_thresholding(data, cov_op=None, basis="haar", thresholding_coeff=0., kind="hard",
                    sing_val=None, sing_vec=None, full: bool=False):
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
    full : bool
        Whether to return the ratios used for thresholding. The default is False.

    Returns
    -------
    thresholded_data : ndarray
        Data after thresholding.
    ratios : ndarray
        The ratios used for thresholding. Returned only if full = True.
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
    ratios = abs(data_in_basis)/np.sqrt(variances)
    ratios = ratios[np.isfinite(ratios)]
    logger.info("Notable values for choosing the thresholding coefficient")
    logger.info("min = %.1g\tmean = %.1g\tmax = %.1g\tmedian = %.1g",
                ratios.min(), ratios.mean(), ratios.max(), np.median(ratios))
    logger.info(f"Thresholded out {mask.sum()} components out of {mask.size}, leaving {mask.size - mask.sum()} components")

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
    if full:
        return thresholded_data, ratios
    else:
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

    def __init__(self, measurement_model):
        super().__init__(measurement_model)
        # Set up the optimization problem
        self._opt_problems = {} # A dictionary of previously solved optimization problems
        # to use for warm start

    def __call__(self, measurement, thresholding_coeff: float=1.,
                 basis: str="eig", skip_tv: bool=False, full: bool=False,
                 downscale_factors=None, use_mask: Optional[bool]=False,
                 warm_start=True,
                 **kwargs) -> np.ndarray:
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
        full : bool
            Whether to return the ratios used for thresholding.
            The default is False.
        downscale_factors : 2-tuple of ints or None, optional
            If not None, downscale the produced image in the specified way.
        use_mask : bool, optional
            If True, consider the values of all pixels with weak transmittance
            to have the same brightness value.
        warm_start : bool
            Whether to use 'warm start', that is, reusing the byproducts
            of solving the previous optimization problem. The default is True.

        Returns
        -------
        numpy.ndarray
            The estimated image.

        """
        t_start = perf_counter()
        try:
            red_res = kwargs["red_res"]
        except KeyError:
            if basis == "eig":
                red_res, sing_val, sing_vec = super().__call__(
                    measurement, eig=True, keep_1d=True,
                    downscale_factors=downscale_factors, use_mask=use_mask,
                )
                if thresholding_coeff > 0:
                    red_res = do_thresholding(red_res, basis=basis,
                                              thresholding_coeff=thresholding_coeff,
                                              sing_val=sing_val, sing_vec=sing_vec,
                                              full=full)
            else:
                if use_mask:
                    raise NotImplementedError(
                        "use_mask == True is not implemented for basis other than the eigenbasis."
                    )
                red_res, cov_op = super().__call__(measurement, calc_cov_op=True,
                                                   downscale_factors=downscale_factors,
                                                   use_mask=use_mask)
                if thresholding_coeff > 0:
                    red_res = do_thresholding(red_res, cov_op, basis=basis,
                                              thresholding_coeff=thresholding_coeff,
                                              full=full)
        if full:
            red_res, remainder = red_res
        if skip_tv:
            red_res = self.to_image(red_res, downscale_factors, use_mask)
            if full:
                return red_res, remainder
            else:
                return red_res
        #TODO Omit the following if sufficient measurement data to estimate the image
        mt_op = self._mt_op(measurement.size, downscale_factors, use_mask)
        if warm_start and (mt_op.shape[0], downscale_factors, use_mask) in self._opt_problems:
            prob, expected_measurement, f = self._opt_problems[(mt_op.shape[0], downscale_factors, use_mask)]
            logger.info(f"Using warm start from {mt_op.shape[0]} measurements")
        else:
            logger.info(f"No previous problem to warm start from for {mt_op.shape[0]} measurements")
            expected_measurement = cp.Parameter(mt_op.shape[0])
            f = cp.Variable(mt_op.shape[1])
            f2 = self.to_image(f, downscale_factors=downscale_factors, use_mask=use_mask)
            sparsity_term = (cp_norm1(cp_diff(f2, k=1, axis=0))
                              + cp_norm1(cp_diff(f2, k=1, axis=1)))**2
            # sparsity_term = cp.atoms.total_variation.tv(f2)**2
            # objective = cp.Minimize(sparsity_term)
            # constraints = [mt_op @ f == expected_measurement]
            # prob = cp.Problem(objective, constraints)
            # In theory, the code above from objective = ... to prob = ...
            # is the correct one. However, CVXPY do not seem to be robust enough
            # to deal with constrained optimization instead of unconstrained one below.
            # In addition, it is faster.
            fidelity = cp.sum((mt_op @ f - expected_measurement)**2)
            #TODO A more intelligent way of picking the alpha value
            # to avoid constrained optimization.
            alpha = 1e-5
            objective = cp.Minimize(fidelity + alpha*sparsity_term)
            prob = cp.Problem(objective)

            self._opt_problems[(mt_op.shape[0], downscale_factors, use_mask)] = (prob, expected_measurement, f)

        expected_measurement.value = mt_op.dot(red_res.ravel())

        #TODO Is there an easier way of trying all available solvers until one succeeds?
        try_solving_until_success(prob, [cp.OSQP, cp.ECOS, cp.SCS], warm_start=warm_start)
        t_end = perf_counter()
        logger.info("Sparse reduction took %.3g s for A shape %s",
                    t_end - t_start, mt_op.shape)
        try:
            result = self.to_image(f.value, downscale_factors, use_mask)
        except AttributeError:
            print("Problem status", prob.status)
            print("Solver name", prob.solver_stats.solver_name)
            print("Problem value", prob.value)
            logger.error("Constrained optimization during measurement reduction did not produce a valid estimate.")
            result = red_res
        if full:
            return result, remainder
        else:
            return result
