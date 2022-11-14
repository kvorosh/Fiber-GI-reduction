# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:14:12 2022

@author: balakin
"""

import numpy as np
import matplotlib.pyplot as plt


def num_to_letter(num: int) -> str:
    """
    Return a letter to be used as a subfigure label.

    Parameters
    ----------
    num : int
        Number of the subfigure, starting with 0.

    Returns
    -------
    letter : str
        The corresponding letter.
    """
    return chr(ord('a') + num)


def for_report_picking_tau():
    img_ids = [3, 2, 6, 7]
    taus = {3: 3.2, 2: 5.6, 6: 6.0, 7: 4.5}

    # plt.figure(figsize=(15/2.54, 9/2.54))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15/2.54, 9/2.54),
                            constrained_layout=True)

    # img_id = img_ids[0]
    for (i, img_id), ax in zip(enumerate(img_ids), axs.flat):
        ratios = np.load(f"tmp-data/ratios-{img_id}.npy")
        ratios.sort()


        # How many components are zeroed out
        ratios2 = np.empty(shape=(2*ratios.size - 1,), dtype=float)
        n_0_comps = np.arange(1, ratios.size + 1)
        n_0_comps2 = np.zeros_like(ratios2)
        ratios2[0::2] = ratios
        n_0_comps2[0::2] = n_0_comps
        ratios2[1::2] = ratios[1:]
        n_0_comps2[1::2] = n_0_comps[: -1]
        # plt.subplot(2, 2, 1)
        # plt.plot(ratios2, n_0_comps2, "-xb")
        # plt.axvline(taus[img_id], color="r")
        # plt.title("0")
        # plt.subplot(2, 2, i + 1)
        # plt.loglog(ratios2, n_0_comps2)
        ax.semilogx(ratios2, n_0_comps2)
        # ax.plot(ratios2, n_0_comps2)
        ax.axvline(taus[img_id], color="r")
        ax.set_title(f"({num_to_letter(i)})")
        ax.set_xlabel("τ")
        ax.set_ylabel("m")

    # plt.savefig(args, kwargs)
    plt.show()


if __name__ == "__main__":
    for_report_picking_tau()
