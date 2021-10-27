# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:18:56 2021

@author: balakin
"""

import logging
from typing import Optional

import cvxpy as cp
from cvxpy.atoms.affine.sum import sum as cp_sum
import numpy as np

from measurement_model import GIProcessingMethod

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
