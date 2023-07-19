# -*- coding: utf-8 -*-
"""
Code related to calculating the result of propagation in an optical fiber.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyMMF
from joblib import Memory
from misc import load_demo_image

memory = Memory("./cachedir", verbose=1)

 # μm for length units
PRESET_0 = dict(areaSize=2.5*50./2, n1=1.45, NA=0.22, a=50./2, id=0)
PRESET_1 = dict(areaSize=3.5*31.25, n1=1.4613, NA=0.275, a=31.25, id=1)


class _Propagator():
    def __init__(self, eigenvectors, prop_matrix):
        self._eigenvectors = eigenvectors
        self._prop_matrix = prop_matrix

    def __call__(self, img):
        coeffs = self._eigenvectors.dot(np.sqrt(img).ravel())
        coeffs = self._prop_matrix.dot(coeffs)
        return (abs(self._eigenvectors.conj().T.dot(coeffs))**2).reshape(img.shape)


@memory.cache
def propagator(grid_dim=64, preset=PRESET_0):
    fiber_length = 50*1e3 # μm
    npoints = grid_dim
    # 1.5 min for 64 points, 5.5 min for 128
    wavelength = 0.6328 # μm

    img_shape = (npoints, npoints)
    profile_params_gen = dict(npoints=npoints, areaSize=preset['areaSize'])
    profile_params_parab = dict(n1=preset["n1"], a=preset["a"], NA=preset["NA"])
    solver_params = dict(curvature=None, propag_only=True, boundary="close")

    profile = pyMMF.IndexProfile(**profile_params_gen)
    profile.initParabolicGRIN(**profile_params_parab)

    solver = pyMMF.propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wavelength)

    n_modes_max = pyMMF.estimateNumModesGRIN(wavelength, profile_params_parab["a"],
                                             profile_params_parab["NA"])

    n_calc_modes = n_modes_max + 10

    modes = solver.solve(mode="eig", nmodesMax=n_calc_modes, **solver_params)

    prop_matrix = modes.getPropagationMatrix(fiber_length)

    # eigenvectors = np.array([modes.profiles[m] for m in range(n_calc_modes)])
    eigenvectors = np.array(modes.profiles)
    # Shape of `eigenvectors`: (n_calc_modes, npoints**2)

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
        if np.iscomplexobj(img):
            img2 = img
        else:
            img2 = np.sqrt(img)
        coeffs = eigenvectors.dot(img2.ravel())
        coeffs = prop_matrix.dot(coeffs)
        return (abs(eigenvectors.conj().T.dot(coeffs))**2).reshape(img_shape)

    return _Propagator(eigenvectors, prop_matrix)


def main():
    from time import perf_counter

    t_start = perf_counter()

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

    t_end = perf_counter()
    print(f"Took {t_end - t_start} s")

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_after_fiber)
    plt.show()

if __name__ == "__main__":
    main()

