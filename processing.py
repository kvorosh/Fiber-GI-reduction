# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:26:32 2021

@author: balakin
"""

import logging
from collections import defaultdict
from os import remove
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import lsmr
from skimage.transform import rescale
from tqdm import tqdm

from misc import load_demo_image, save_image_for_show
from reduction import GIDenseReduction, GISparseReduction, GIDenseReductionIter
from measurement_model import GIMeasurementModel, pad_or_trim_to_shape, TraditionalGI
from fiber_propagation import PRESET_0, PRESET_1
from compressive_sensing import (GICompressiveSensingL1DCT,
                                 GICompressiveSensingL1Haar, GICompressiveTC2,
                                 GICompressiveAnisotropicTotalVariation,
                                 GICompressiveAnisotropicTotalVariation2)

logger = logging.getLogger("FGI-red")
logger.propagate = False # Do not propagate to the root logger
logger.setLevel(logging.INFO)
logger.handlers = []
fh = logging.FileHandler("processing.log")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def synth(measurement, mt_op, img_shape, noise_var, omega):
    tmp_op = mt_op.dot(mt_op.T).astype(float)
    tmp_op[np.diag_indices_from(tmp_op)] += noise_var*omega
    return mt_op.T.dot(np.linalg.solve(tmp_op, measurement)).reshape(img_shape)


def compressive_tv(measurement, mt_op, img_shape, alpha=None):
    initial_estimate = lsmr(mt_op, measurement)[0]
    beta = 1.

    def sparsity_func(f):
        diff0 = np.diff(f.reshape(img_shape), axis=0, n=1)[:, :-1]
        diff1 = np.diff(f.reshape(img_shape), axis=1, n=1)[:-1, :]
        smoothed_diffs = np.sqrt(beta**2 + diff0**2 + diff1**2)

        tmp0 = diff0/smoothed_diffs
        tmp1 = diff1/smoothed_diffs
        grad = np.zeros_like(f.reshape(img_shape))
        grad[: -1, : -1] = -tmp0 - tmp1
        grad[1:, : -1] += tmp0
        grad[: -1, 1:] += tmp1
        return smoothed_diffs.sum(), grad.ravel()

    if alpha is None:
        con = LinearConstraint(mt_op, measurement, measurement)
        res = minimize(sparsity_func, initial_estimate, constraints=con,
                       jac=True, options={"disp": True})
    else:
        def combined_obj_func(f):
            residual = measurement - mt_op.dot(f)
            main_val = (residual**2).sum()
            main_grad = -2*mt_op.T.dot(residual)
            sparsity_val, sparsity_grad = sparsity_func(f)
            return main_val + alpha*sparsity_val, main_grad + alpha*sparsity_grad

        res = minimize(combined_obj_func, initial_estimate, jac=True, options={"disp": True})

    return res.x.reshape(img_shape)


def figure_name_format(img_id, n_patterns, noise_var=0., kind="", alpha=None,
                       other_params=None, meas_mod_desc=""):
    name = f"{img_id}_{meas_mod_desc}_{n_patterns}_{noise_var:.0e}_{kind}"
    if alpha is not None:
        if alpha != "":
            try:
                name = name + "_{:.0e}".format(alpha)
            except TypeError:
                name = name + "_{}".format(alpha)
    if other_params is not None:
        name = name + "_{}".format(other_params)
    return name


def prepare_measurements(data_source=3, noise_var: float = 0, n_patterns: int = 1024,
                         pattern_type: str="pseudorandom", img_shape=None,
                         fiber_opts=PRESET_0, n_measurements: int=1, sensor_fovs=None):
    if not isinstance(data_source, int):
        measurement = data_source
        n_patterns = measurement.size
    if '*' in pattern_type:
        # pixel_size = 5.2 # μm
        pixel_size = 1. # pixel
    else:
        img_shape = (128, 128)
        pixel_size = (2.5 * 50. / 4)/img_shape[0]

    model = GIMeasurementModel(
        n_patterns, img_shape, pattern_type, pixel_size=pixel_size,
        fiber_opts=fiber_opts, sensor_fovs=sensor_fovs
    )

    if isinstance(data_source, int): # pylint: disable=R1705
        src_img = load_demo_image(data_source)
        if n_measurements > 1:
            measurement = [model.simulate_measurement(src_img, noise_var)
                           for _ in range(n_measurements)]
        else:
            measurement = model.simulate_measurement(src_img, noise_var)
    return measurement, model


def finding_alpha(img_id: int = 3, noise_var: float = 0, proc_kind: str = "l1") -> None:
    """
    Generate images of processing results by the specified method for some
    values or regularization factor alpha to determine the best value.

    Parameters
    ----------
    img_id : int, optional
        Id of the image of the object to be used for processing. The default is 3.
    noise_var : float, optional
        Noise variance. The default is 0.
    proc_kind : str, optional
        Specification of the processing method. The default is "l1".
        The valid values are "l1", "l1h", "tc2", "tva" and "tva2".

    Returns
    -------
    None
    """

    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var
    )
    processing_method = {m.name: m(model) for m in [GICompressiveSensingL1DCT,
                                                    GICompressiveSensingL1Haar,
                                                    GICompressiveTC2,
                                                    GICompressiveAnisotropicTotalVariation,
                                                    GICompressiveAnisotropicTotalVariation2]}

    if not isinstance(img_id, int):
        img_id = None

    for alpha in [1e-9, 1e-6, 1e-3, 1e-1, 1, 1e1]:
        estimate = processing_method[proc_kind](measurement, alpha=alpha)
        save_image_for_show(
            estimate, figure_name_format(img_id, noise_var, proc_kind,
                                         alpha=alpha, meas_mod_desc=model.suffix),
            rescale=True
        )
    logger.info("Done for imd_id = %s, noise_var = %.3g and proc_kind = %s",
                img_id, noise_var, proc_kind)


def finding_alpha_l_curve(img_id: int = 3, noise_var: float = 0,
                          proc_kind: str = "l1", n_measurements: int = 1024) -> None:
    """
    Save data to plot the L-curve for finding the value of alpha
    to use in image processing.

    Parameters
    ----------
    img_id : int, optional
        Id of the image of the object to be used for processing. The default is 3.
    noise_var : float, optional
        Noise variance. The default is 0.
    proc_kind : str, optional
        Specification of the processing method. The default is "l1".
        The valid values are "l1", "l1h", "tc2", "tva" and "tva2".
    n_measurements : int
        The number of illumination patterns to use.

    Returns
    -------
    None

    """
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var
    )
    processing_method = {m.name: m(model)
                         for m in [GICompressiveSensingL1DCT,
                                   GICompressiveSensingL1Haar,
                                   GICompressiveTC2,
                                   GICompressiveAnisotropicTotalVariation,
                                   GICompressiveAnisotropicTotalVariation2]}[proc_kind]

    if not isinstance(img_id, int):
        img_id = None

    residuals = []
    reg_terms = []
    alpha_values = np.geomspace(1e-5, 1e2, num=11)
    #TODO Try a parallel execution of the loop body
    # Though it will likely fail due to lack of memory
    for alpha in tqdm(alpha_values):
        _, resid, reg = processing_method(measurement, alpha=alpha, full=True)
        residuals.append(resid)
        reg_terms.append(reg)

    residuals = np.array(residuals)
    reg_terms = np.array(reg_terms)
    np.savez_compressed(
        "../l_curve/{}_{:.0e}_{}_{}.npz".format(
            img_id, noise_var, proc_kind, n_measurements
        ),
        alpha=alpha_values, resid=residuals, reg=reg_terms
    )


def plot_l_curve(img_id: int = 3, noise_var: float = 0, proc_kind: str = "l1",
                 n_measurements: int = 1024) -> None:
    """
    Plot the L-curve for finding the value of alpha to use in image processing.

    Parameters
    ----------
    img_id : int, optional
        Id of the image of the object to be used for processing. The default is 3.
    noise_var : float, optional
        Noise variance. The default is 0.
    proc_kind : str, optional
        Specification of the processing method. The default is "l1".
        The valid values are "l1", "l1h", "tc2", "tva" and "tva2".
    n_measurements : int
        The number of illumination patterns to use.

    Returns
    -------
    None

    """
    if not isinstance(img_id, int):
        img_id = None
    with np.load("../l_curve/{}_{:.0e}_{}_{}.npz".format(
            img_id, noise_var, proc_kind, n_measurements
    )) as data:
        alpha_values = data["alpha"]
        reg_terms = data["reg"]
        residuals = data["resid"]
    plt.loglog(reg_terms, residuals)
    plt.xlabel("regularity")
    plt.ylabel("residual")
    for x, y, alpha in zip(reg_terms, residuals, alpha_values):
        plt.annotate("a = {:.3g}".format(alpha), (x, y), xycoords="data")
    plt.show()


def finding_iter_params(img_id: int = 3, noise_var: float = 0) -> None:
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var
    )
    if not isinstance(img_id, int):
        img_id = None

    for relax in [1.]:
        if relax == 1.:
            pp = []
        else:
            pp = False
        estimate = GIDenseReductionIter(model)(measurement, relax=relax,
                                               n_iter=1000000, print_progress=pp)
        save_image_for_show(estimate, figure_name_format(
            img_id, noise_var, "red-iter", alpha=relax, meas_mod_desc=model.suffix
        ), rescale=True)
        if pp:
            plt.plot(pp)
    logger.info("Done for imd_id = %s, noise_var = %.3g and proc_kind = red-iter",
                img_id, noise_var)
    plt.show()


def show_methods(img_id=3, noise_var=0., n_patterns=1024,
                 save: bool=True, show: bool=True,
                 pattern_type: str="pseudorandom", fiber_opts=PRESET_0) -> None:
    t_start = perf_counter()
    if '*' in pattern_type:
        # img_shape = (105, 105)
        img_shape = (200, 200)
    else:
        img_shape = None
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var,
        n_patterns=n_patterns,
        pattern_type=pattern_type,
        img_shape=img_shape,
        fiber_opts=fiber_opts
    )

    logger.info("Setting up took %.3g s", perf_counter() - t_start)

    if isinstance(img_id, int):
        src_img = pad_or_trim_to_shape(load_demo_image(img_id), model.img_shape).astype(float)
    else:
        #TODO Think of a better identifier
        img_id = pattern_type.split('/')[1]
        src_img = None

    estimates = {}

    t_estim_part = perf_counter()
    estimates[TraditionalGI.name] = TraditionalGI(model)(measurement)
    logger.info("Traditional GI formation took %.3g s.",
                perf_counter() - t_estim_part)

    # whitened_measurement = noise_var**0.5 * measurement

    alpha_values = {("tc2", 3, 1e-1): 6e-3, ("tva2", 3, 1e-1): 0.158,
                    ("tc2", 2, 1e-1): 1e-3, ("tva2", 2, 1e-1): 0.158,
                    ("tc2", 6, 1e-1): 6e-3,
                    ("tc2", 7, 1e-1): 1e-3}

    alpha_values = defaultdict(lambda: 1e-5, alpha_values)
    # None, corresponding to alpha = 0+ would be a more accurate default value,
    # especially for no noise,
    # but this provides the same results faster.


    cs_processing_methods = [
        GICompressiveSensingL1DCT,
        GICompressiveSensingL1Haar,
        GICompressiveTC2,
        GICompressiveAnisotropicTotalVariation,
        GICompressiveAnisotropicTotalVariation2
    ]

    for processing_method in cs_processing_methods:
        t_estim_start = perf_counter()
        estimates[processing_method.name] = processing_method(model)(
            measurement,
            alpha=alpha_values[(processing_method.name, img_id, float(noise_var))]
        )
        logger.info("Estimation using %s took %.3g s.", processing_method.name,
                    perf_counter() - t_estim_start)

    # tau_values = {(2, 0.0): 1.0, (2, 0.1): 1,
    #               (3, 0.): 1e-05, (3, 0.1): 0.1,
    #               (6, 0.): 1.0, (6, 0.1): 1,
    #               (7, 0.): 10.0, (7, 0.1): 1.,
    #               (8, 0.): 1e-05, (8, 0.1): 1e-5}
    tau_values = {(2, 0.0): 1.0, (2, 0.1): 5.6,
                    (3, 0.): 1e-05, (3, 0.1): 3.7,
                    (6, 0.): 1.0, (6, 0.1): 5.3,
                    (7, 0.): 10.0, (7, 0.1): 4.3,
                    (8, 0.): 1e-05, (8, 0.1): 1e-5}
    tau_values = defaultdict(lambda: 1., tau_values)

    processing_methods = [TraditionalGI] + cs_processing_methods + [GIDenseReduction, GISparseReduction]

    estimates[GIDenseReduction.name] = GIDenseReduction(model)(measurement)
    estimates[GISparseReduction.name] = GISparseReduction(model)(
        measurement, tau_values[(img_id, noise_var)], basis="eig"
    )
    t_end = perf_counter()
    logger.info("show_methods for %d patterns and %s shape took %.3g s",
                measurement.size, src_img.shape, t_end - t_start)

    if save:
        if src_img is not None:
            save_image_for_show(src_img, figure_name_format(
                img_id, n_patterns, noise_var, "src", "",
                meas_mod_desc=model.suffix
            ), unit_scale=True)
        save_image_for_show(estimates[TraditionalGI.name],
                            figure_name_format(
                                img_id, n_patterns, noise_var, TraditionalGI.name,
                                "", meas_mod_desc=model.suffix
                           ), rescale=True)
        for cs_method in cs_processing_methods:
            save_image_for_show(
                estimates[cs_method.name], figure_name_format(
                    img_id, n_patterns, noise_var, cs_method.name,
                    alpha=alpha_values[(cs_method.name, img_id, float(noise_var))],
                    meas_mod_desc=model.suffix
                ), rescale=True
            )
        save_image_for_show(estimates[GIDenseReduction.name], figure_name_format(
            img_id, n_patterns, noise_var, GIDenseReduction.name, "",
            meas_mod_desc=model.suffix
        ), rescale=True)
        save_image_for_show(estimates[GISparseReduction.name], figure_name_format(
            img_id, n_patterns, noise_var, GISparseReduction.name,
            tau_values[(img_id, float(noise_var))],
            meas_mod_desc=model.suffix
        ), rescale=True)

    if show:
        subplot_no = 0
        size = model.pixel_size * model.img_shape[0]/2

        def plot_part(image, part_title):
            nonlocal subplot_no
            subplot_no += 1
            plt.subplot(3, 3, subplot_no)
            plt.imshow(image, cmap=plt.cm.gray, #pylint: disable=E1101
                       extent=[-size, size, -size, size])
            plt.xlabel("x, " + model.unit)
            plt.ylabel("y, " + model.unit)
            plt.title(part_title)

        fig = plt.gcf()
        # fig.clear()
        fig.set_tight_layout(True)

        if src_img is not None:
            plot_part(src_img, "Объект исследования")
        for method in processing_methods:
            name = method.name
            desc = method.desc
            plot_part(estimates[name], desc)
    # mng = plt.get_current_fig_manager()
    # try:
    #     mng.frame.Maximize(True)
    # except AttributeError:
    #     mng.window.showMaximized()
    # img_names = {2: "phys", 3: "two_slits", 6: "teapot128", 7: "teapot64"}
    # plt.savefig("../figures/{}_{}.pdf".format(
    #     img_names[img_id], "noisy" if noise_var > 0 else "noiseless"
    # ))
        plt.show()


def for_report_picking_tau_data(img_id=3, noise_var=0., n_measurements=1024,
                           pattern_type: str="pseudorandom", fiber_opts=PRESET_0) -> None:
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var, n_patterns=n_measurements,
        pattern_type=pattern_type,
        img_shape=(128, 128),
        fiber_opts=fiber_opts
    )

    measurement -= measurement.mean()

    estimator = GISparseReduction(model)
    result, ratios = estimator(measurement, thresholding_coeff=1.,
                       basis="eig", full=True, skip_tv=True)
    np.save(f"tmp-data/ratios-{img_id}.npy", ratios)


def for_report_intermediate_results(img_id, skip_thr=False):
    noise_var = 0.1
    n_measurements = 1024
    pattern_type = "pseudorandom-phase"

    logger.info(f"Started calculating intermediate results for img_id = {img_id}")
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var,
        n_patterns=n_measurements,
        pattern_type=pattern_type,
        img_shape=(128, 128),
        fiber_opts=PRESET_1
    )
    # measurement -= measurement.mean()
    logger.info("Done simulating measurements")

    tau_values = {(2, 0.0): 1.0, (2, 0.1): 5.6,
                    (3, 0.): 1e-05, (3, 0.1): 3.7,
                    (6, 0.): 1.0, (6, 0.1): 1.,
                    (7, 0.): 10.0, (7, 0.1): 1.}
    tau_values = defaultdict(lambda: 1., tau_values)


    fname = f"tmp-data/intermediate-{img_id}.npz"
    estimator_sparse = GISparseReduction(model)
    try:
        frames = dict(np.load(fname))
    except FileNotFoundError:
        frames = {}
    if "direct" not in frames:
        frames["direct"] = estimator_sparse(
            measurement, tau_values[(img_id, noise_var)], basis="eig"
        )
        np.savez(fname, **frames)
    red_linear = GIDenseReduction(model)
    red_linear_iter = GIDenseReductionIter(model)
    relax = 0.3#0.15
    n_cycles = 1024
    # 1 cycle is measurement.size iterations
    _, sing_val, sing_vec = red_linear(measurement, eig=True)
    interm_iter_nos = 2**np.arange(0, 21, 3)
    result_lin_iter, interm = red_linear_iter(measurement, n_iter=n_cycles*measurement.size,
                                              return_cond=lambda n: n in interm_iter_nos,
                                              relax=relax)
    if interm[-1][0] != n_cycles*measurement.size:
        interm.append((n_cycles*measurement.size, result_lin_iter))
    logger.info("Done Kaczmarz's method")
    from reduction import do_thresholding
    for iter_no, interm_result in interm:
        key = "iter" + str(iter_no)
        if key in frames:
            continue
        logger.info(f"Starting optimization for iteration {iter_no}")
        interm_result = do_thresholding(interm_result, basis="eig",
                                        thresholding_coeff=tau_values[(img_id, noise_var)],
                                        sing_val=sing_val, sing_vec=sing_vec)
        frames[key] = estimator_sparse(measurement, tau_values[(img_id, noise_var)],
                                       red_res=interm_result)
        logger.info(f"Done optimization for iteration {iter_no}")
        np.savez(fname, **frames)
    logger.info(f"Finished calculating intermediate results for img_id = {img_id}")


def for_report_intermediate_lowres(img_id):
    noise_var = 0.1
    n_measurements = 1024
    pattern_type = "pseudorandom-phase"

    logger.info(f"Started calculating intermediate results for img_id = {img_id}")
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var,
        n_patterns=n_measurements,
        pattern_type=pattern_type,
        img_shape=(128, 128),
        fiber_opts=PRESET_1
    )
    # measurement -= measurement.mean()
    logger.info("Done simulating measurements")

    tau_values = {(2, 0.0): 1.0, (2, 0.1): 5.6,
                    (3, 0.): 1e-05, (3, 0.1): 3.7,
                    (6, 0.): 1.0, (6, 0.1): 1.,
                    (7, 0.): 10.0, (7, 0.1): 1.}
    tau_values = defaultdict(lambda: 1., tau_values)
    factors = (2, 2)

    fname = f"tmp-data/interm-lowres-{img_id}.npz"
    estimator_sparse = GISparseReduction(model)
    estimator_iter = GIDenseReductionIter(model)
    relax = 0.3#0.15
    estimator_lin = GIDenseReduction(model)

    try:
        with np.load(fname) as d:
            data = dict(d)
    except FileNotFoundError:
        frames = {}
    if "direct" not in frames:
        frames["direct"] = estimator_sparse(
            measurement, tau_values[(img_id, noise_var)], basis="eig"
        )
        np.savez(fname, **frames)

    from reduction import do_thresholding

    if "lr" in frames: # Low-resolution intermediate estimate
        result_lin_iter = frames["lr"]
    else:
        n_cycles = 64#1024
        _, sing_val, sing_vec = estimator_lin(measurement, eig=True, downscale_factors=factors)
        result_lin_iter = estimator_iter(measurement, n_iter=n_cycles*measurement.size,
                                                 relax=relax, downscale_factors=factors)
        result_lin_iter = do_thresholding(result_lin_iter, basis="eig",
                                          thresholding_coeff=tau_values[(img_id, noise_var)],
                                          sing_val=sing_val, sing_vec=sing_vec)
        result_lin_iter = estimator_sparse(measurement, tau_values[(img_id, noise_var)],
                                           red_res=result_lin_iter, downscale_factors=factors)
        frames["lr"] = result_lin_iter
        np.savez(fname, **frames)
    if "lru" in frames: # Low-resolution intermediate estimate, upscaled
        interm_hr = frames["lru"]
    else:
        interm_hr = rescale(result_lin_iter, factors)
        frames["lru"] = interm_hr
        np.savez(fname, **frames)

    _, sing_val, sing_vec = estimator_lin(measurement, eig=True)
    n_cycles = 512
    if "iter-hr" not in frames:
        result_final = estimator_iter(measurement, n_iter=n_cycles*measurement.size,
                                      relax=relax, start_from=interm_hr)
        # logger.info(f"Starting optimization for iteration {iter_no}")
        result_final = do_thresholding(result_final, basis="eig",
                                        thresholding_coeff=tau_values[(img_id, noise_var)],
                                        sing_val=sing_val, sing_vec=sing_vec)
        frames["iter-hr"] = estimator_sparse(measurement, tau_values[(img_id, noise_var)],
                                        red_res=result_final)
        # logger.info(f"Done optimization for iteration {iter_no}")
        np.savez(fname, **frames)


def for_report_fiber_mask(img_id):
    noise_var = 0.1
    n_measurements = 1024
    pattern_type = "pseudorandom-phase"

    logger.info(f"Started calculating results for masking for img_id = {img_id}")
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var,
        n_patterns=n_measurements,
        pattern_type=pattern_type,
        img_shape=(128, 128),
        fiber_opts=PRESET_1
    )
    # measurement -= measurement.mean()
    logger.info("Done simulating measurements")

    tau_values = {(2, 0.0): 1.0, (2, 0.1): 5.6,
                    (3, 0.): 1e-05, (3, 0.1): 3.7,
                    (6, 0.): 1.0, (6, 0.1): 1.,
                    (7, 0.): 10.0, (7, 0.1): 1.}
    tau_values = defaultdict(lambda: 1., tau_values)

    fname = f"tmp-data/fiber-masking-{img_id}.npz"
    estimator_sparse = GISparseReduction(model)

    try:
        with np.load(fname) as d:
            data = dict(d)
    except FileNotFoundError:
        data = {}
    if "direct" not in data:
        data["direct"] = estimator_sparse(measurement,
                                          tau_values[(img_id, noise_var)])
        np.savez(fname, **data)
    if "mask" not in data:
        mask = model.fiber_mask
        data["mask"] = mask
        np.savez(fname, **data)
    if "mask-direct" not in data:
        estim = estimator_sparse(measurement, tau_values[(img_id, noise_var)],
                                 use_mask=True)
        data["mask-direct"] = estim
        np.savez(fname, **data)
    logger.info(f"Finished calculating results for masking for img_id = {img_id}")


def for_article_multiple_sensors(img_id):
    noise_var = 0.1
    # n_measurements = 1024
    n_measurements_all = {3: 128, 2: 256, 6: 128, 7: 128}
    n_measurements = n_measurements_all[img_id]
    pattern_type = "pseudorandom-phase"

    center_offset = int(np.floor(37/2**0.5))
    radius = center_offset
    center = np.array((64, 64))

    def fill_circle(arr, circle_center, circle_radius):
        idx = np.indices(arr.shape)
        dist_sq = (idx[0] - circle_center[0])**2 + (idx[1] - circle_center[1])**2
        arr[dist_sq <= circle_radius**2] = 1

    fovs = np.zeros((4, 128, 128), dtype=bool)
    fill_circle(fovs[0, ...], (center[0] + center_offset, center[1]), radius)
    fill_circle(fovs[1, ...], (center[0] - center_offset, center[1]), radius)
    fill_circle(fovs[2, ...], (center[0], center[1] + center_offset), radius)
    fill_circle(fovs[3, ...], (center[0], center[1] - center_offset), radius)

    logger.info(f"Started calculating results for multiple detectors for img_id = {img_id}")
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var,
        n_patterns=n_measurements,
        pattern_type=pattern_type,
        img_shape=(128, 128),
        fiber_opts=PRESET_1,
        sensor_fovs=fovs
    )
    # pattern shape: (1024, 128, 128) before applying the FoV data
    # and (4096, 128, 128) afterwards
    measurement2, model2 = prepare_measurements(
        img_id, noise_var=noise_var,
        n_patterns=4*n_measurements,
        pattern_type=pattern_type,
        img_shape=(128, 128),
        fiber_opts=PRESET_1
    )
    # measurement -= measurement.mean()
    logger.info("Done simulating measurements")


    tau_values = {(2, 0.0): 1.0, (2, 0.1): 5.6,
                    (3, 0.): 1e-05, (3, 0.1): 3.7,
                    (6, 0.): 1.0, (6, 0.1): 1.,
                    (7, 0.): 10.0, (7, 0.1): 1.}
    tau_values = defaultdict(lambda: 1., tau_values)

    fname = f"tmp-data/mult-sens-{img_id}.npz"
    estimator_sparse = GISparseReduction(model)
    estimator_sparse2 = GISparseReduction(model2)

    try:
        with np.load(fname) as d:
            data = dict(d)
    except FileNotFoundError:
        data = {}
    if "direct" not in data:
        data["direct"] = estimator_sparse2(measurement2[: n_measurements],
                                          tau_values[(img_id, noise_var)])
        np.savez(fname, **data)
    if "direct2" not in data:
        data["direct2"] = estimator_sparse2(measurement2[: 2*n_measurements],
                                          tau_values[(img_id, noise_var)])
        np.savez(fname, **data)
    if "direct4" not in data:
        data["direct4"] = estimator_sparse2(measurement2,
                                          tau_values[(img_id, noise_var)])
        np.savez(fname, **data)
    if "mask" not in data:
        mask = model.fiber_mask
        data["mask"] = mask
        np.savez(fname, **data)
    else:
        mask = data["mask"]
    if "fovs" not in data:
        data["fovs"] = fovs
        np.savez(fname, **data)
    if "mask-direct" not in data:
        estim_mask = estimator_sparse2(measurement2[: n_measurements],
                                       tau_values[(img_id, noise_var)],
                                       use_mask=True)
        data["mask-direct"] = estim_mask
        np.savez(fname, **data)
    if "mask-direct2" not in data:
        estim_mask = estimator_sparse2(measurement2[: 2*n_measurements],
                                       tau_values[(img_id, noise_var)],
                                       use_mask=True)
        data["mask-direct2"] = estim_mask
        np.savez(fname, **data)
    if "mask-direct4" not in data:
        estim_mask = estimator_sparse2(measurement2,
                                       tau_values[(img_id, noise_var)],
                                       use_mask=True)
        data["mask-direct4"] = estim_mask
        np.savez(fname, **data)
    else:
        estim_mask = data["mask-direct4"]
    if "fov-mask" not in data:
        fov_estim = estimator_sparse(measurement, tau_values[(img_id, noise_var)],
                                     use_mask=True)
        data["fov-mask"] = fov_estim
        np.savez(fname, **data)
    else:
        fov_estim = data["fov-mask"]
    logger.info(f"Finished calculating results for masking for img_id = {img_id}")


def show_single_method(img_id=3, noise_var=0., n_measurements=1024, pattern_type: str="pseudorandom") -> None:
    measurement, model = prepare_measurements(
        img_id, noise_var=noise_var, n_patterns=n_measurements,
        pattern_type=pattern_type,
        img_shape=(200, 200)
    )

    # if isinstance(img_id, int):
    #     src_img = pad_or_trim_to_shape(load_demo_image(img_id), model.img_shape)

    # "dct", no noise: 1e-5
    # "dct", 1e-2 noise: at least 1

    # basis = "eig"
    # # thr_coeff_values = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 1.]
    # # thr_coeff_values = [10, 100, 1e3, 1e4, 1e5]
    # thr_coeff_values = [5e-4, 5e-3]
    # for thr_coeff in thr_coeff_values:
    #     result = sparse_reduction(measurement, mt_op, img_shape,
    #                           thresholding_coeff=thr_coeff, basis=basis)
    #     diff_sq = np.linalg.norm(result - src_img)**2
    #     save_image_for_show(result.clip(0, None), "red_sparse_{}_{:.0e}_{}_{:.0e}".format(
    #         img_id, noise_var, basis, thr_coeff
    #     ), rescale=True)
    #     with open("red_sparse_diff.txt", "a", encoding="utf-8") as f:
    #         f.write("{}\t{:.1g}\t{}\t{:.1g}\t{:.3g}\n".format(img_id, noise_var, basis, thr_coeff, diff_sq))

    # result = GICompressiveAnisotropicTotalVariation2(model)(measurement, alpha=1e-5)
    # # print(np.linalg.norm(result - src_img)**2)

    # result = GISparseReduction(model)(measurement, thresholding_coeff=0.1,
    #                                    basis="eig")
    # print(np.linalg.norm(result - src_img)**2)

    estimator = GIDenseReductionIter(model)
    result = estimator(measurement, n_iter=100000)

    estimator_bis = GISparseReduction(model)
    result_bis = estimator_bis(measurement, 1e6, skip_tv=False)
    # result_bis = estimator_bis(measurement, 5e6, skip_tv=False)
    # result_bis = estimator_bis(measurement, 1e8, skip_tv=True)

    result2 = TraditionalGI(model)(measurement)

    plt.subplot(131)
    plt.imshow(result, cmap=plt.cm.gray) # pylint: disable=E1101
    plt.title("Линейная редукция измерения")
    plt.subplot(132)
    plt.imshow(result_bis, cmap=plt.cm.gray) # pylint: disable=E1101
    plt.title("Редукция измерения, предлагаемый метод")
    plt.subplot(133)
    plt.imshow(result2, cmap=plt.cm.gray) # pylint: disable=E1101
    plt.title("Обычное ФИ")

    plt.show()


def se_calculations(img_id: int=3, noise_var: float=0.1, tau_value: float=0.1,
                    pattern_type: str="pseudorandom") -> None:

    output = "../data/se_{}_{}_{:.0e}_{:.0e}.dat".format(img_id, pattern_type[0], noise_var, tau_value)
    intermediate_results = "../data/_se_{}_{}_{:.0e}_{:.0e}.dat".format(img_id, pattern_type[0], noise_var, tau_value)
    max_n_patterns = 13000
    total_measurement, model = prepare_measurements(
        img_id, noise_var=noise_var, n_patterns=max_n_patterns,
        pattern_type=pattern_type
    )
    src_img = pad_or_trim_to_shape(load_demo_image(img_id), model.img_shape)

    se_results = []
    header = ["n", "gi",
              # "l1",
              "tva", "tva2", "reds"]

    t_start = perf_counter()

    if pattern_type == "quasirandom":
        n_patterns_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 288, 320, 352, 368,
                             384, 400, 416, 448, 480, 512, 1024, 2048]
    else:
        n_patterns_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 288, 320, 352, 368,
                             384, 400, 416, 448, 480, 512, 1024, 2048, 4096, 8192,
                             max_n_patterns]
    se_results = np.zeros((len(n_patterns_values), len(header)))
    header = '\t'.join(header)
    try:
        prev_results = np.loadtxt(intermediate_results)
    except OSError:
        start_ind = 0
    else:
        start_ind = prev_results.shape[0]
        se_results[: start_ind, :] = prev_results
        n_patterns_values = n_patterns_values[start_ind: ]

    for i, n_patterns in tqdm(enumerate(n_patterns_values, start_ind),
                              total=len(n_patterns_values)):
        measurement = total_measurement[: n_patterns]
        se_results[i, 0] = n_patterns

        # Traditional ghost imaging
        traditional_gi = TraditionalGI(model)(measurement)
        se_results[i, 1] = np.linalg.norm(traditional_gi - src_img)**2

        # L1 compressive sensing
        # cs_l1 = compressive_l1(measurement, mt_op, src_img.shape, alpha=1e-5)
        # current_results.append(np.linalg.norm(cs_l1 - src_img)**2)

        # anisotropic TV compressive sensing
        cs_tva = GICompressiveAnisotropicTotalVariation(model)(measurement, alpha=1e-5)
        se_results[i, 2] = np.linalg.norm(cs_tva - src_img)**2

        # anisotropic TV compressive sensing, second version
        cs_tva2 = GICompressiveAnisotropicTotalVariation2(model)(measurement, alpha=1e-5)
        se_results[i, 3] = np.linalg.norm(cs_tva2 - src_img)**2

        # measurement reduction
        reds = GISparseReduction(model)(measurement, thresholding_coeff=tau_value)
        se_results[i, 4] = np.linalg.norm(reds - src_img)**2

        np.savetxt(intermediate_results, se_results[: i + 1, ...],
                   delimiter='\t', fmt='%.5g', header=header)

    t_end = perf_counter()
    logger.info("Calculations took {:.3g} s.".format(t_end - t_start))

    np.savetxt(output.format(img_id, noise_var), se_results,
               delimiter='\t', fmt='%.5g', header=header)
    try:
        remove(intermediate_results)
    except FileNotFoundError:
        pass
if __name__ == "__main__":
    # show_single_method(3, 0)
    # show_single_method(3, 1e-1)
    # show_single_method(3, 1e-1, 1024)
    # show_single_method(6, 0)
    # show_single_method(6, 1e-1)
    # show_single_method(7, 0)
    # show_single_method(7, 1e-1)
    # show_methods(8, 1e-1, n_patterns=512, save=True, show=True, pattern_type="speckle")
    # show_methods(3)
    # show_methods(3, 1e-1)
    # show_methods(2)
    # show_methods(2, 1e-1)
    # show_methods(6)
    # show_methods(6, 1e-1)
    # show_methods(7)
    # show_methods(7, 1e-1)
    # show_single_method(np.load("bucket_data/objest_data_TDC_10_11.npy").astype(float),
    #                     pattern_type="speckle_patterns/quantum_11_10_v1/slm*.bmp")
    # show_single_method(np.load("bucket_data/objest_data_TDC_11_12.npy").astype(float),
                        # pattern_type="speckle_patterns/quantum_11_12/slm*.bmp")
    # show_methods(6, 1e-1, n_patterns=1024, save=False, show=True,
    #               pattern_type="pseudorandom-phase", fiber_opts=PRESET_1)
    # for_report_picking_tau_data(3, 1e-1, 1024, "pseudorandom-phase", PRESET_1)
    # for_report_picking_tau_data(2, 1e-1, 1024, "pseudorandom-phase", PRESET_1)
    # for_report_picking_tau_data(6, 1e-1, 1024, "pseudorandom-phase", PRESET_1)
    # for_report_picking_tau_data(7, 1e-1, 1024, "pseudorandom-phase", PRESET_1)
    # for img_id in [3, 2, 6, 7]:
    #     for_report_intermediate_results(img_id)
    # finding_alpha(3, 0., "l1")
    # finding_alpha(3, 0., "l1h")
    # finding_alpha(3, 0., "tc2")
    # finding_alpha(3, 0., "tva")
    # finding_alpha(3, 1e-1, "l1")
    # finding_alpha(3, 1e-1, "l1h")
    # finding_alpha(3, 1e-1, "tc2")
    # finding_alpha(3, 1e-1, "tva")
    # finding_alpha(2, 0., "l1")
    # finding_alpha(2, 0., "l1h")
    # finding_alpha(2, 0., "tc2")
    # finding_alpha(2, 0., "tva")
    # finding_alpha(2, 1e-1, "l1")
    # finding_alpha(2, 1e-1, "l1h")
    # finding_alpha(2, 1e-1, "tc2")
    # finding_alpha(2, 1e-1, "tva")
    # finding_alpha(6, 0., "l1")
    # finding_alpha(6, 0., "l1h")
    # finding_alpha(6, 0., "tc2")
    # finding_alpha(6, 0., "tva")
    # finding_alpha(6, 1e-1, "l1")
    # finding_alpha(6, 1e-1, "l1h")
    # finding_alpha(6, 1e-1, "tc2")
    # finding_alpha(6, 1e-1, "tva")
    # finding_alpha(7, 0., "tc2")
    # finding_alpha(7, 0., "tva")
    # finding_alpha(7, 1e-1, "tc2")
    # finding_alpha(7, 1e-1, "tva")
    # finding_alpha(7, 0., "l1")
    # finding_alpha(7, 0., "l1h")
    # finding_alpha(7, 1e-1, "l1")
    # finding_alpha(7, 1e-1, "l1h")
    # finding_alpha_l_curve(3, 1e-1, "l1")
    # finding_alpha_l_curve(3, 1e-1, "tva")
    # finding_alpha_l_curve(6, 1e-1, "l1")
    # finding_alpha_l_curve(6, 1e-1, "l1h")
    # finding_alpha_l_curve(6, 1e-1, "tc2")
    # finding_alpha_l_curve(6, 1e-1, "tva")
    # finding_alpha_l_curve(6, 1e-1, "tva2")
    # finding_alpha_l_curve(7, 1e-1, "l1")
    # finding_alpha_l_curve(7, 1e-1, "l1h")
    # finding_alpha_l_curve(7, 1e-1, "tc2")
    # finding_alpha_l_curve(7, 1e-1, "tva")
    # for img_id in [3, 2, 6, 7]:
    #     for_report_intermediate_lowres(img_id)
    # for img_id in [3, 2, 6, 7]:
    #     for_report_fiber_mask(img_id)
    for img_id in [3, 2, 6, 7]:
        for_article_multiple_sensors(img_id)
    # show_methods(7, 1e-1, n_patterns=512,
    #               pattern_type="speckle_patterns/11_22_2021_spekls_1cm/slm*.bmp",
    #               save=True, show=True)
    # show_methods(np.load("bucket_data/obj_data_nol_3mm_5k.npy").astype(float),
    #                     pattern_type="speckle_patterns/11_24_2021_spekls_3mm_nol/slm*.bmp", show=True)
    # finding_alpha_l_curve(7, 1e-1, "tva2")
    # plot_l_curve(3, 1e-1, "l1")
    # finding_iter_params(3, 0.)
