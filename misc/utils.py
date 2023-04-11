from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TKAgg')


def find_file(relative_dir, file_name):
    res = []
    for path in Path(relative_dir).rglob(file_name):
        res.append(path)

    if len(res) > 0:
        return str(res[0])

    return ""


def f(img, alpha=2.0, beta=-200):
    alpha = np.array([alpha], dtype=np.uint16)

    img = np.clip((img * alpha) + beta, 0, 255).astype(np.uint8)
    return img


class DebugContext:
    def __init__(self, name, active=True):
        self._name = name
        self._is_active = active

    @property
    def name(self):
        return self._name

    @property
    def is_active(self):
        return self._is_active


def show_image(imgs, cmap='gray', cols=2, dbg_ctx=None):
    # if dbg_ctx is not None:
    if dbg_ctx is None or not dbg_ctx.is_active:
        return

    n_imgs = len(imgs)
    rows = (n_imgs // cols) + 1

    if isinstance(imgs, tuple):
        img1, title = imgs
        plt.title(title)
        plt.imshow(img1, cmap=cmap)
        plt.yticks([])
        plt.xticks([])

        plt.show()
        return

    if isinstance(imgs, np.ndarray):
        img1 = imgs
        plt.yticks([])
        plt.xticks([])
        plt.imshow(imgs, cmap=cmap)

        plt.show()
        return

    for i, img in enumerate(imgs):
        img1 = None
        title = ""

        if isinstance(img, tuple) and len(img) == 2:
            img1, title = img
        else:
            img1 = img

        plt.subplot(rows, cols, i + 1)
        plt.title(title)
        plt.yticks([])
        plt.xticks([])
        plt.imshow(img1, cmap=cmap)

    plt.subplots_adjust(bottom=0.0, right=0.679, left=0.364, top=0.871, wspace=0.0, hspace=0.052)
    plt.show()
