# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:18:56 2021

@author: balakin
"""

import logging
from typing import Optional

import cvxpy as cp
from cvxpy.atoms import norm as cp_norm
from cvxpy.atoms.affine.diff import diff as cp_diff
from cvxpy.atoms.affine.reshape import reshape as cp_reshape
from cvxpy.atoms.affine.sum import sum as cp_sum
from cvxpy.atoms.norm1 import norm1 as cp_norm1
import numpy as np
from scipy.fft import dctn, idctn  # pylint: disable=E0611

from measurement_model import GIProcessingMethod
from haar_transform import haar_transform, inverse_haar_transform_2d

logger = logging.getLogger("Fiber-GI-reduction.cs")


class GICompressiveSensing(GIProcessingMethod):
    """
    An abstract class for various compressive sensing methods used
    to process the measurement data into the ghost image based on them.

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

    def _regularization_term(self, estimate, **kwargs):
        raise NotImplementedError("Not implemented for the abstract class!")

    def _postprocess(self, estimate) -> np.ndarray:
        return estimate.reshape(self._measurement_model.img_shape)

    def __call__(self, measurement, # pylint: disable=W0221
                 alpha: Optional[float]=None, full: bool=False, **kwargs) -> np.ndarray:
        """
        Estimate the image using the specified compressive sensing approach.

        Parameters
        ----------
        measurement : array_like
            The measurement. If the measurement size is less than
            the available number of patterns, only the first
            `measurement.size` ones are used.
        alpha: float or None
            The regularization coefficient value. None corresponds to
            minimization of the regularization term under the constraint
            of recovering the provided measurement data.
            Otherwise, the processing is done by minimization of the least
            squares fidelity term and the multiplied regularization term.
            The default is None.
        full : bool, optional
            Whether to return additional information (the residual norm and
            the value of the regularization term). The default is False.

        Returns
        -------
        numpy.ndarray
            The processing result.

        """
        mt_op = self._mt_op(measurement.size)
        estimate = cp.Variable(mt_op.shape[1])
        sparsity = self._regularization_term(estimate, **kwargs)
        if alpha is None:
            constraints = [mt_op @ estimate == measurement]
            objective = cp.Minimize(estimate)
            prob_constr = cp.Problem(objective, constraints)
            prob_constr.solve()
        else:
            # TODO Compare below expression with
            # cp_norm(measurement - self._measurement_model.mt_op @ estimate, 2)
            fidelity = cp_sum((measurement - mt_op @ estimate)**2)
            objective = cp.Minimize(fidelity + alpha*sparsity)
            prob = cp.Problem(objective)
            prob.solve()
            fidelity = fidelity.value
            sparsity = sparsity.value
        result = self._postprocess(estimate.value)
        if alpha is not None and full: # pylint: disable=R1705
            return result, fidelity, sparsity
        else:
            return result


class GICompressiveSensingL1DCT(GICompressiveSensing):
    """
    Compressive sensing for the case when the regularization term
    is the L1 norm of the image in DCT basis.

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
    name = "l1"
    desc = "Сжатые измерения, мин. нормы L1 в базисе DCT"

    def _mt_op(self, n_patterns: Optional[int]=None) -> np.ndarray:
        mt_op = super()._mt_op(n_patterns)
        tmp_op = mt_op.T.reshape(self._measurement_model.img_shape + (-1,))
        mt_op_dct = dctn(tmp_op, axes=(0, 1), norm="ortho").reshape( # pylint: disable=E1101
            (self._measurement_model.img_shape[0]*self._measurement_model.img_shape[1], -1)
        ).T
        return mt_op_dct

    def _regularization_term(self, estimate):
        return cp_norm1(estimate)**2

    def _postprocess(self, estimate) -> np.ndarray:
        estimate = super()._postprocess(estimate)
        estimate = idctn(estimate, axes=(0, 1), norm="ortho")
        return estimate


class GICompressiveSensingL1Haar(GICompressiveSensing):
    """
    Compressive sensing for the case when the regularization term
    is the L1 norm of the image in Haar basis.

    Note
    ----
    Image dimensions have to be powers of 2.

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
    name = "l1h"
    desc = "Сжатые измерения, мин. нормы L1 в базисе Хаара"

    def _mt_op(self, n_patterns: Optional[int]=None) -> np.ndarray:
        mt_op = super()._mt_op(n_patterns)
        tmp_op = mt_op.reshape((-1,) + self._measurement_model.img_shape)
        back_transform_matrix = haar_transform(np.eye(self._measurement_model.img_shape[0]), axis=0,
                                               inplace=True)
        mt_op_haar = np.tensordot(tmp_op, back_transform_matrix, axes=(1, 0))
        if self._measurement_model.img_shape[1] != self._measurement_model.img_shape[0]:
            back_transform_matrix = haar_transform(
                np.eye(self._measurement_model.img_shape[1]), axis=0, inplace=True
            )
        mt_op_haar = np.tensordot(mt_op_haar, back_transform_matrix, axes=(1, 0))
        mt_op_haar = mt_op_haar.reshape((-1, self._measurement_model.mt_op.shape[1]))
        return mt_op_haar

    def _regularization_term(self, estimate):
        return cp_norm1(estimate)**2

    def _postprocess(self, estimate) -> np.ndarray:
        estimate = super()._postprocess(estimate)
        estimate = inverse_haar_transform_2d(estimate)
        return estimate


class GICompressiveTC2(GICompressiveSensing):
    """
    Compressive sensing for the case when the regularization term
    is total squared curvature of the image.

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
    name = "tc2"
    desc = "Сжатые измерения, мин. полной квадратичной кривизны"

    def _regularization_term(self, estimate):
        estimate = cp_reshape(estimate, self._measurement_model.img_shape)
        sparsity_term = (cp_norm(cp_diff(estimate, k=2, axis=0), 2)
                         + cp_norm(cp_diff(estimate, k=2, axis=1), 2))
        return sparsity_term


class GICompressiveAnisotropicTotalVariation(GICompressiveSensing):
    """
    Compressive sensing for the case when the regularization term
    is the L1 norm version of anisotropic total variation.

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
    name = "tva"
    desc = "Сжатые измерения, мин. анизотропного вар-та вариации"

    def _regularization_term(self, estimate):
        estimate = cp_reshape(estimate, self._measurement_model.img_shape)
        sparsity_term = (cp_norm1(cp_diff(estimate, k=1, axis=0))
                         + cp_norm1(cp_diff(estimate, k=1, axis=1)))**2
        return sparsity_term


class GICompressiveAnisotropicTotalVariation2(GICompressiveSensing):
    """
    Compressive sensing for the case when the regularization term
    is the L2 norm version of anisotropic total variation.

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
    name = "tva2"
    desc = "Сжатые измерения, мин. альт. анизотропного вар-та вариации"

    def _regularization_term(self, estimate):
        estimate = cp_reshape(estimate, self._measurement_model.img_shape)
        sparsity_term = cp.atoms.total_variation.tv(estimate)**2
        return sparsity_term
