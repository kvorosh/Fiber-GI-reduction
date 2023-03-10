# -*- coding: utf-8 -*-
"""
Construction of the measurement model describing the formation
of the acquired data.
"""

import logging
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple

# cached_property is in the standard library from Python 3.8
try:
    from functools import cached_property
# If it isn't available, try the following package
except ImportError:
    from cached_property import cached_property

import numpy as np
from imageio import imread
from scipy.stats.qmc import Sobol
from skimage.transform import downscale_local_mean
from tqdm import tqdm
from cvxpy.atoms.affine.reshape import reshape as cp_reshape

from fiber_propagation import propagator
from misc import apply_mask_to_mt_op, transform_using_mask

logger = logging.getLogger("FGI-red.measmodel")

#TODO Add an attribute to store noise variance information?
class GIMeasurementModel:
    """
    The mathematical model of how the measurement data from the bucket detector
    relate to the studied object.

    Parameters
    ----------
    n_patterns : int
        The number of illumination patterns, corresponding to measurement size.
    img_shape : 2-tuple of ints or None, optional
        The shape of the ghost image. Can be omitted if the illumination patterns
        are loaded from files instead of calculations. In that case if it is not None,
        the loaded patterns are resized to approximately given size
        ('approximately' because since skimage.transform.downscale_local_mean is used).
    pattern_type : {"pseudorandom", "quasirandom", "pseudorandom-phase",
                    "quasirandom-phase", "speckle"}, optional
        What illumination patterns to use. Valid values are "pseudorandom",
        "quasirandom", "pseudorandom-phase", "quasirandom-phase"
        (corresponding to binary or random-phase pseudo- or quasirandom patterns
        which then pass through the optical fiber) and "speckle" (corresponding
        to acquired photos of illumination patterns). Photos are loaded from
        the files 'speckle_patterns/slmX', where X are integers starting from 0
        and the format is any supported by `imageio.imread`.
        The default is "pseudorandom".
    pixel_size : float or None, optional
        Image pixel size in e.g. metric units.
    unit : str or None, optional
        The unit of pixel_size.
    fiber_opts : dict or None, optional
        Dictionary of properties of the optical fiber. For examples, see
        PRESET_0 and PRESET_1 in fiber_propagation.py

    Attributes
    ----------
    mt_op : numpy.ndarray
        The matrix of the linear operator whose rows are raveled illumination
        patterns. Has shape (n_patterns, img_shape[0]*img_shape[1]).
    img_shape : 2-tuple of ints
        The shape of the ghost image.
    pixel_size : float, optional
        Image pixel size in e.g. metric units. In the future, it will be
        a mandatory argument if illumination patterns are calculated,
        in which case it has to be measured in μm.
    unit : str, optional
        The unit of pixel_size.
    suffix : str
        Short descriptor for illumination pattern origin.
        Empty before initialization. After initialization, the first letter
        may be 'p' (pseudorandom patterns), 'q' (quasirandom patterns)
        or 'c' (patterns are loaded instead of being generated).
        If patterns are generated,
        the second letter may be 'b' (binary patterns before fiber)
        or 'p' (random phase patterns before fiber).
        The remaining letters are ID of fiber_options argument.
    """

    def __init__(self, n_patterns: int, img_shape: Optional[Tuple[int, int]]=None, # pylint: disable=R0913
                 pattern_type: str="pseudorandom", pixel_size: float=1.,
                 unit: str="px", fiber_opts=None):
        self.n_patterns = n_patterns
        self.pixel_size = pixel_size
        self.unit = unit
        self._fiber_opts = fiber_opts
        logger.info("Using optical fiber options %s", self._fiber_opts)
        self.suffix = ""
        if "*" in pattern_type:
            illum_patterns = self._load_speckle_patterns(img_shape, pattern_type)
            self.img_shape = illum_patterns.shape[1:]
            self.suffix = "c"
        else:
            if img_shape is None:
                raise ValueError(
                    "Must provide the image shape unless using speckle patterns."
                )
            self.img_shape = img_shape
            if pattern_type == "pseudorandom":
                illum_patterns = self._pseudorandom_patterns()
                self.suffix = "pb" + str(fiber_opts["id"])
            elif pattern_type == "quasirandom":
                illum_patterns = self._quasirandom_patterns()
                self.suffix = "qb" + str(fiber_opts["id"])
            elif pattern_type == "pseudorandom-phase":
                illum_patterns = self._pseudorandom_patterns(phase=True)
                self.suffix = "pp" + str(fiber_opts["id"])
            elif pattern_type == "quasirandom-phase":
                illum_patterns = self._quasirandom_patterns(phase=True)
                self.suffix = "qp" + str(fiber_opts["id"])
        self.mt_op = illum_patterns.reshape((self.n_patterns, -1))

    def mt_op_part(self, n_patterns: Optional[int]=None) -> np.ndarray:
        """
        Extract a part of `mt_op` corresponding to first `n_patterns` measurements.

        Parameters
        ----------
        n_patterns : int or None, optional
            The number of patterns. The default is None.

        Returns
        -------
        numpy.ndarray of shape (n_patterns, img_shape[0]*img_shape[1])
            The extracted part of mt_op.

        """
        if n_patterns is None:
            return self.mt_op
        return self.mt_op[: n_patterns, ...]

    @cached_property
    def fiber_mask(self):
        """
        Returns a mask whose False elements correspond to pixels whose brightnesses
        are not transmitted over the fiber well enough.

        Returns
        -------
        bool numpy.ndarray of shape `img_shape`.
            The mask.

        """
        threshold = 1e-5 # The relative threshold
        t_start = perf_counter()
        logger.info("Started calculating the mask")
        propagate_func = propagator(self.img_shape[0], self._fiber_opts)
        white_img = np.ones(self.img_shape)
        mask = propagate_func(white_img) >= threshold
        t_end = perf_counter()
        logger.info("Finished calculating the mask, took  %.3g s", t_end - t_start)
        logger.info("nnz ratio = %d / %d", np.count_nonzero(mask), np.size(mask))
        return mask

    def illumination_patterns(self, n_patterns: int=None) -> np.ndarray:
        """
        Reshapes the mt_op so that mt_op[i, ...] are the illumination patterns.

        Parameters
        ----------
        n_patterns : int or None, optional
            The number of illumination patterns to return. If None, all patterns
            are returned.

        Returns
        -------
        numpy.ndarray of shape (n_patterns, img_shape[0], img_shape[1])
            The illumination patterns.

        """
        return self.mt_op[: n_patterns, ...].reshape((-1,) + self.img_shape)

    def _pseudorandom_patterns(self, phase=False):
        rng = np.random.default_rng(2021)
        #TODO Allow for non-square images
        #TODO pass self.img_shape and self.pixel_size to pyMMF calculations
        propagate_func = propagator(self.img_shape[0], self._fiber_opts)
        if phase:
            logger.info("Preparing %d pseudorandom phase patterns", self.n_patterns)
            illum_patterns = rng.random(size=(self.n_patterns,) + self.img_shape).astype(float)*2*np.pi
            logger.info("Prepared %d pseudorandom phase patterns", self.n_patterns)
        else:
            illum_patterns = rng.integers(0, 1, size=(self.n_patterns,) + self.img_shape,
                                          endpoint=True).astype(float)
        for i in range(self.n_patterns):
            if phase:
                img = np.exp(1j*illum_patterns[i, ...])
            else:
                img = illum_patterns[i, ...]
            illum_patterns[i, ...] = propagate_func(img)
        return illum_patterns

    def _quasirandom_patterns(self, phase=False):
        propagate_func = propagator(self.img_shape[0], self._fiber_opts)

        gen = Sobol(
            self.img_shape[0]*self.img_shape[1], scramble=False, seed=2021
        ).fast_forward(1)
        illum_patterns = gen.random(self.n_patterns).reshape(
            (self.n_patterns,) + self.img_shape
        )
        if phase:
            illum_patterns *= 2*np.pi
        else:
            illum_patterns = (illum_patterns >= 0.5).astype(float)

        for i in range(self.n_patterns):
            if phase:
                img = np.exp(1j*illum_patterns[i, ...])
            else:
                img = illum_patterns[i, ...]
            illum_patterns[i, ...] = propagate_func(img)
        return illum_patterns

    def _load_speckle_patterns(self, img_shape, pattern_template):
        logger.info("Loading %d illumination patterns from %s",
                    self.n_patterns, pattern_template)
        t_start = perf_counter()
        ref_dir = Path(".")
        pattern_list = [p for p in ref_dir.glob(pattern_template) if p.is_file()][: self.n_patterns]
        for pattern_no, pattern_path in enumerate(tqdm(pattern_list)):
            raw_pattern = imread(pattern_path, as_gray=True)
            # if raw_pattern.shape[0] == 430 and raw_pattern.shape[1] == 430:
            #     crop_to = [0, 400, 0, 400]
            #     raw_pattern = raw_pattern[crop_to[2]: crop_to[3], crop_to[0]: crop_to[1]]
            if img_shape is not None:
                #TODO Make it so the produced size is *not larger* than the specified
                #TODO Calculate the factors only once
                old_shape = np.array(raw_pattern.shape)
                img_shape = np.array(img_shape)
                factors = tuple(np.rint(old_shape/img_shape).astype(int))
                if pattern_no == 0:
                    logger.info("Original size is %s", old_shape)
                    logger.info("Trying to reduce size to %s", img_shape)
                    logger.info("Binning factors are %s", factors)
                raw_pattern = downscale_local_mean(raw_pattern, factors)
            if pattern_no == 0:
                logger.info("Pattern shape is %s", raw_pattern.shape)
                illum_patterns = np.empty((self.n_patterns,) + raw_pattern.shape, dtype=float)
            illum_patterns[pattern_no, ...] = raw_pattern
        t_end = perf_counter()
        logger.info("Loading illumination patterns took %.3g s.", t_end - t_start)
        return illum_patterns

    @cached_property
    def _noise_rng(self):
        return np.random.default_rng(2021) # seed is set for repeatability

    def simulate_measurement(self, source_image,
                             noise_var: float=0.) -> np.ndarray:
        """
        Simulate the measurement process for a given image of the studied object.

        Parameters
        ----------
        source_image : array_like of shape img_shape
            The image of the studied object.
        noise_var : float, optional
            Variance of the white noise to be added to the measurement.
            The default is 0..

        Returns
        -------
        measurement : ndarray of shape (n_patterns,)
            The simulated measurement

        """
        source_image = pad_or_trim_to_shape(source_image, self.img_shape)

        measurement = self.mt_op.dot(source_image.ravel())

        if noise_var > 0:
            measurement = measurement.astype(float)
            measurement += self._noise_rng.normal(
                scale=noise_var**0.5, size=measurement.shape
            )
        return measurement


def pad_or_trim_to_shape(img, shape: Tuple[int, int]) -> np.ndarray:
    """
    Pad an image with zeros or trim it, in both cases keeping the image
    as centered as possible, to make it the specified shape.

    Parameters
    ----------
    img : array_like
        The image.
    shape : 2-tuple of int
        The shape.

    Returns
    -------
    img : ndarray
        The adjusted image.

    """
    if shape[0] == img.shape[0] and shape[1] == img.shape[1]:
        return img
    if (shape[0] != img.shape[0]
        or shape[1] != img.shape[1]):
        to_add_0 = shape[0] - img.shape[0]
        to_add_1 = shape[1] - img.shape[1]
        if to_add_0 < 0:
            diff = -to_add_0//2
            img = img[diff: diff + shape[0], :]
            to_add_0 = 0
        if to_add_1 < 0:
            diff = -to_add_1//2
            img = img[:, diff: diff + shape[1]]
            to_add_1 = 0
        img = np.pad(img,
                     ((to_add_0//2, to_add_0 - to_add_0//2),
                      (to_add_1//2, to_add_1 - to_add_1//2)),
                     mode="constant", constant_values=0.)
    return img


class GIProcessingMethod:
    """
    An abstract class for various methods used to process the measurement data
    into the ghost image based on them.

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
    name = ""
    desc = ""

    def __init__(self, model: GIMeasurementModel):
        self._measurement_model = model

    def _mt_op(self, n_patterns: Optional[int]=None, downscale_factors=None,
               use_mask=False) -> np.ndarray:
        mt_op = self._measurement_model.mt_op_part(n_patterns)
        if downscale_factors is not None:
            factors = (1,) + downscale_factors
            mt_op = mt_op.reshape((-1, ) + self._measurement_model.img_shape)
            mt_op = downscale_local_mean(mt_op, factors)
            mt_op = mt_op.reshape((mt_op.shape[0], -1))
        if use_mask:
            mask = self._measurement_model.fiber_mask
            if downscale_factors is not None:
                # If any of the pixels to be 'merged' is True, the result should also be True.
                mask = downscale_local_mean(mask, factors) > 0
            mt_op = apply_mask_to_mt_op(mt_op, mask)
        return mt_op

    def __call__(self, measurement, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Not implemented in the general case!")

    def img_shape(self, downscale_factors=None):
        """
        The shape of the output image.

        Parameters
        ----------
        downscale_factors : 2-tuple of ints or None, optional
            If not None, downscale the produced image in the specified way.
            The default is None.

        Returns
        -------
        None.

        """
        orig_shape = self._measurement_model.img_shape
        if downscale_factors is None:
            return orig_shape
        else:
            return (orig_shape[0]//downscale_factors[0], orig_shape[1]//downscale_factors[1])

    def to_image(self, data, downscale_factors=None, use_mask: bool=False) -> np.ndarray:
        """
        Convert a 1D array into the corresponding image. Equivalent to reshaping
        if use_mask is False.

        Parameters
        ----------
        data : array_like
            The data for reshaping.
        downscale_factors : 2-tuple of ints, optional
            If not None, the produced image considered to be downscaled
            in the specified way. The default is None.
        use_mask : bool, optional
            If True, the last element of `data` is considered
            to hold the value of all pixels for which the corresponding mask element
            is False.
            The default is False.

        Returns
        -------
        image : numpy.ndarray
            The corresponding image.
        """
        if use_mask:
            mask = self._measurement_model.fiber_mask
            if downscale_factors is not None:
                # If any of the pixels to be 'merged' is True, the result should also be True.
                mask = downscale_local_mean(mask, downscale_factors) > 0
            transform = transform_using_mask(mask)
            data = transform @ data
        try:
            image = data.reshape(self.img_shape(downscale_factors))
        except AttributeError:
            image = cp_reshape(data, self.img_shape(downscale_factors))
        return image


class TraditionalGI(GIProcessingMethod):
    """
    Ghost image formation using the traditional approach, that is,
    summation of the illumination patterns weighted by the corresponding
    measurement components after subtracting the mean.

    Parameters
    ----------
    model : GIMeasurementModel
        The ghost image measurement model on which to base the processing.
    """
    name = "gi"
    desc = "Обычное ФИ"

    def __call__(self, measurement) -> np.ndarray: # pylint: disable=W0221
        """
        Process the measurement using the traditional approach, that is,
        summation of the illumination patterns weighted by the corresponding
        measurement components after subtracting the mean.
        If the measurement is shorter than the available number of patterns,
        only the first `measurement.size` ones are used.

        Parameters
        ----------
        measurement : array_like
            The measurement.

        Returns
        -------
        result : numpy.ndarray
            The processing result.
        """
        illum_patterns = self._measurement_model.illumination_patterns(measurement.size)
        result = np.tensordot(measurement - measurement.mean(),
                              illum_patterns - illum_patterns.mean(axis=0),
                              axes=1)/measurement.size
        return result
