# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:17:59 2023

@author: balakin
"""

import logging
# from time import perf_counter
from os.path import splitext, basename
import matplotlib.pyplot as plt
import numpy as np
from reduction import GIDenseReduction, GISparseReduction, GIDenseReductionIter, do_thresholding
from measurement_model import GIMeasurementModel, TraditionalGI


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


def choice_of_tau(glob_to_patterns: str, path_to_measurement: str, n_measurements: int|None=None) -> None:
    measurement = np.load(path_to_measurement)
    print("Total of", measurement.size, "measurements.")
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
                  n_measurements: int|None=None, tau_value: float=1.,
                  disp: bool=True, direct: bool=True) -> np.ndarray:
    measurement = np.load(path_to_measurement)
    if n_measurements is None:
        n_measurements = measurement.size
    else:
        measurement = measurement[: n_measurements]

    model = GIMeasurementModel(n_measurements, pattern_type=glob_to_patterns)

    estimator = GISparseReduction(model)

    if direct:
        estimate = estimator(measurement, tau_value,
                              skip_tv=n_measurements>=model.img_shape[0]*model.img_shape[1])
    else:
        estimator_k = GIDenseReductionIter(model)
        n_cycles = 64
        estimate_lin = estimator_k(measurement, n_iter=n_cycles*n_measurements)
        _, sing_val, sing_vec = GIDenseReduction(model)(measurement, eig=True)
        result_interm = do_thresholding(estimate_lin, basis="eig", thresholding_coeff=3e2,
                                    sing_val=sing_val, sing_vec=sing_vec)
        estimate = estimator(measurement, tau_value, red_res=result_interm,
                             skip_tv=n_measurements>=model.img_shape[0]*model.img_shape[1])

    if disp:
        plt.imshow(estimate, cmap=plt.cm.gray)
        plt.show()

    return estimate


def main(glob_to_patterns: str, path_to_measurement: str, tau: float, n_measurements: list, direct: bool=True):
    estimates = [do_estimation(glob_to_patterns, path_to_measurement, n, tau, False, direct)
                 for n in n_measurements]
    name = "red_" + ("" if direct else "iter_") + splitext(basename(path_to_measurement))[0]
    np.savez(f"figures/{name}.npz",
              **{f"red_{nmeas}": est
                for nmeas, est in zip(n_measurements, estimates)})

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(19.2/2.54, 9.8/2.54))
    axs = axs.ravel()

    for ax, n, u in zip(axs, n_measurements, estimates):
        ax.imshow(u, cmap=plt.cm.gray)
        ax.axis("off")
        ax.set_title(f"{n} измерений")

    fig.savefig(f"figures/{name}.pdf")
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # choice_of_tau("speckle_patterns/patt_ob7_6k/pat*.bmp", "bucket_data/obj2_tdc_1bin_6k_final.npy", 6000)
    # do_estimation("speckle_patterns/patt_ob7_6k/pat*.bmp", "bucket_data/obj2_tdc_1bin_6k_final.npy",
    #               2000, 3e1)
    main("speckle_patterns/patt_ob7_6k/pat*.bmp", "bucket_data/obj2_tdc_1bin_6k_final.npy",
          3e2, [600, 1000, 1500, 2400, 3800, 6000], False)
