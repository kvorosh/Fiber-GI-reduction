# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:17:59 2023

@author: balakin
"""

import logging
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from misc import load_demo_image, save_image_for_show
from reduction import GIDenseReduction, GISparseReduction, GIDenseReductionIter
from measurement_model import GIMeasurementModel, pad_or_trim_to_shape, TraditionalGI
from compressive_sensing import (GICompressiveSensingL1DCT,
                                 GICompressiveSensingL1Haar, GICompressiveTC2,
                                 GICompressiveAnisotropicTotalVariation,
                                 GICompressiveAnisotropicTotalVariation2)


logger = logging.getLogger("FGI-red")
logger.propagate = False # Do not propagate to the root logger
logger.setLevel(logging.INFO)
logger.handlers = []
fh = logging.FileHandler("processing2.log")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def choice_of_tau(glob_to_patterns, path_to_measurement, n_measurements: int|None=None) -> None:
    measurement = np.load(path_to_measurement)
    if n_measurements is None:
        n_measurements = measurement.size
    else:
        measurement = measurement[: n_measurements]

    model = GIMeasurementModel(n_measurements, pattern_type=glob_to_patterns)
    estimator = GISparseReduction(model)
    _, ratios = estimator(measurement, 1., basis="eig", full=True, skip_tv=True)
    ratios.sort()
    ratios2 = np.empty(shape=(2*ratios.size - 1,), dtype=float)
    n_0_comps = np.arange(1, ratios.size + 1)
    n_0_comps2 = np.zeros_like(ratios2)
    ratios2[0::2] = ratios
    n_0_comps2[0::2] = n_0_comps
    ratios2[1::2] = ratios[1:]
    n_0_comps2[1::2] = n_0_comps[: -1]
    plt.semilogx(ratios2, n_0_comps2)
    plt.xlabel("τ")
    plt.ylabel("m")
    plt.show()


def do_estimation(glob_to_patterns: str, path_to_measurement: str,
                  n_measurements: int|None=None, disp: bool=True) -> np.ndarray:
    measurement = np.load(path_to_measurement)
    if n_measurements is None:
        n_measurements = measurement.size
    else:
        measurement = measurement[: n_measurements]

    model = GIMeasurementModel(n_measurements, pattern_type=glob_to_patterns)

    estimator = GISparseReduction(model)

    estimate = estimator(measurement, 3e2,
                         skip_tv=n_measurements>=model.img_shape[0]*model.img_shape[1])


    if disp:
        plt.imshow(estimate)
        plt.show()

    return estimate


def main():
    n_measurements = [400, 700, 1000, 2000, 5000, 8000]

    estimates = [do_estimation("speckle_patterns/For_BDA_obj2/slm*.bmp",
                               "bucket_data/obj2_1bin_8k_v3.npy", n, False)
                 for n in n_measurements]

    for i, (n, u) in enumerate(zip(n_measurements, estimates), 1):
        plt.subplot(2, 3, i)
        plt.imshow(u, cmap=plt.cm.gray)
        plt.title(f"{n} измерений")

    plt.show()


if __name__ == "__main__":
    # do_estimation("speckle_patterns/For_BDA_obj2/slm*.bmp", "bucket_data/obj2_1bin_8k_v3.npy", 4000)
    # choice_of_tau("speckle_patterns/For_BDA_obj2/slm*.bmp", "bucket_data/obj2_1bin_8k_v3.npy", 2000)
    main()
