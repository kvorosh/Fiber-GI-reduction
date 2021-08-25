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
