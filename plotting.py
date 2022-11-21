# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:14:12 2022

@author: balakin
"""

import numpy as np
import matplotlib.pyplot as plt
from measurement_model import pad_or_trim_to_shape
from misc import load_demo_image


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


def for_report_intermediate_lowres():
    img_ids = [3, 2, 6, 7]
    key_seq = ["lr", "lru", "iter-hr"]

    nrows = len(key_seq)
    ncols = len(img_ids)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(15/2.54, 8/2.54), constrained_layout=True)
    if ncols == 1:
        axs = axs.reshape((-1, 1))

    for col_no, img_id in enumerate(img_ids):
        fname = f"tmp-data/interm-lowres-{img_id}.npz"
        with np.load(fname) as data:
            imgs = {key: data[key] for key in key_seq}
        for row_no, key in enumerate(key_seq):
            ax = axs[row_no, col_no]
            ax.imshow(imgs[key], cmap=plt.cm.gray)
            ax.axis("off")
            ax.set_title(f"({num_to_letter(col_no + row_no*ncols)})")

    plt.savefig("figures/low_resolution_intermediate.pdf", bbox_inches="tight")
    plt.show()


def for_report_intermediate():
    img_ids = [3, 2, 6, 7]

    nrows = 6
    ncols = len(img_ids)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            # figsize=(16/2.54, ncols*8/2.54),
                            figsize=(15/2.54, 15/2.54),
                            constrained_layout=True)
    if ncols == 1:
        axs = axs.reshape((-1, 1))

    for col_no, img_id in enumerate(img_ids):
        fname = f"tmp-data/intermediate-{img_id}.npz"

        data = np.load(fname)
        frame_nos = [int(k[4:]) for k in data.keys() if k != "direct"]
        frame_nos.sort()
        for k in data.keys():
            img_shape = data[k].shape
            break
        src_img = pad_or_trim_to_shape(load_demo_image(img_id), img_shape).astype(float)

        ax = axs[0, col_no]
        ax.imshow(src_img, cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_title(f"({num_to_letter(col_no)})")
        # frames_to_show = (2**np.arange(11) * 1024)[1::3]
        # frames_to_show = 2**np.arange(0, 4, 1)
        frames_to_show = 2**np.array([12, 15, 18, 20])
        print(frames_to_show)
        for i, no in enumerate(frames_to_show):
            res_thr = data["iter" + str(no)]
            ax = axs[i + 1, col_no]
            ax.imshow(res_thr, cmap=plt.cm.gray)
            ax.axis('off')
            if i == 0:
                ax.set_title(f"({num_to_letter(col_no + ncols)})")
            # else:
            #     ax.set_title(str(no) + " it.")
        ax = axs[1 + len(frames_to_show), col_no]
        ax.imshow(data["direct"], cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_title(f"({num_to_letter(col_no + 2*ncols)})")
        data.close()

    plt.savefig("figures/kaczmarz_intermediate.pdf", bbox_inches="tight")

    plt.show()


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
    # for_report_picking_tau()
    # for_report_intermediate()
    for_report_intermediate_lowres()
