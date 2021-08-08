# -*- coding: utf-8 -*-
"""
Code related to calculating the result of propagation in an optical fiber.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyMMF
from misc import load_demo_image


def propagator(grid_dim=64):
    diam = 50. # μm
    fiber_length = 50*1e3 # μm
    npoints = grid_dim
    # 1.5 min for 64 points, 5.5 min for 128
    wavelength = 0.6328 # μm

    img_shape = (npoints, npoints)
    area_size = 2.5 * diam/2
    profile_params_gen = dict(npoints=npoints, areaSize=area_size)
    profile_params_parab = dict(n1=1.45, a=diam/2, NA=0.22)
    solver_params = dict(curvature=None, propag_only=True, boundary="close")

    try:
        with np.load("fiber_properties.npz") as data:
            prop_matrix = data["prop_matrix"]
            eigenvectors = data["eigenvectors"]
    except FileNotFoundError:
        profile = pyMMF.IndexProfile(**profile_params_gen)
        profile.initParabolicGRIN(**profile_params_parab)

        solver = pyMMF.propagationModeSolver()
        solver.setIndexProfile(profile)
        solver.setWL(wavelength)

        n_modes_max = pyMMF.estimateNumModesGRIN(wavelength, diam/2,
                                                 profile_params_parab["NA"])

        n_calc_modes = n_modes_max + 10

        modes = solver.solve(mode="eig", nmodesMax=n_calc_modes, **solver_params)

        prop_matrix = modes.getPropagationMatrix(fiber_length)

        # eigenvectors = np.array([modes.profiles[m] for m in range(n_calc_modes)])
        eigenvectors = np.array(modes.profiles)
        # Shape of `eigenvectors`: (n_calc_modes, npoints**2)
        np.savez_compressed("fiber_properties", prop_matrix=prop_matrix, eigenvectors=eigenvectors)

    def propagate_image(img):
        """
        Calculate the image at fiber output given its input.
        The image is presumed to be detected in a phase-insensitive way.

        Parameters
        ----------
        img : array_like
            The image at the input of the fiber.

        Returns
        -------
        ndarray
            The image at the output of the array.
        """
        coeffs = eigenvectors.dot(np.sqrt(img).ravel())
        coeffs = prop_matrix.dot(coeffs)
        return (abs(eigenvectors.T.dot(coeffs))**2).reshape(img_shape)

    return propagate_image


def main():
    npoints = 128
    img_shape = (npoints, npoints)
    img = np.zeros(img_shape)
    img_central = load_demo_image(3, full_span=True)
    dim_diff_0 = npoints - img_central.shape[0]
    dim_diff_1 = npoints - img_central.shape[1]
    img[dim_diff_0//2: dim_diff_0//2 + img_central.shape[0],
        dim_diff_1//2: dim_diff_1//2 + img_central.shape[1]] = img_central

    propagate_func = propagator(npoints)

    img_after_fiber = propagate_func(img)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_after_fiber)
    plt.show()

if __name__ == "__main__":
    main()

