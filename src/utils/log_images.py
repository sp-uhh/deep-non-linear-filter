import torch
from torchvision.utils import make_grid
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Union

def make_image_grid(image_batch: torch.Tensor, vmin: Union[float, None], vmax: Union[float, None], num_images: int =
-1, n_img_per_row: int = 8):
    """
    Take a batch of 2D data and create a grid of visualizations.

    :param image_batch: the 2D data [BATCH, XAXIS, YAXIS]
    :param num_images: the number of images to be displayed (default: full batch)
    :param vmin, vmax: the min and max value in the data (e.g. vmin=0, vmax=1 for a Wiener-like mask). If vmin and
    vmax are None, they will be inferred from the data. Be aware that the color range might change between epochs then.
    :param n_img_per_row: number of images per row
    """

    image_batch = image_batch[:num_images if num_images > 0 else len(image_batch)].cpu().detach().numpy()

    def rgba_to_rgb(rgba):
        """
        Converts a numpy RGBA array with shape [rows, columns, channels=rgba] to a numpy RGB array [rows, columns,
        channels=rgb] assuming a black background.

        :param rgba: the rgba data in a numpy array
        :return: a numpy array with the same shape as the input array but with the number of channels reduced to 3
        """
        a = rgba[..., 3]
        rgba[..., 0] = a * rgba[..., 0] + (1 - a) * 255
        rgba[..., 1] = a * rgba[..., 1] + (1 - a) * 255
        rgba[..., 2] = a * rgba[..., 2] + (1 - a) * 255

        return rgba[..., :3]

    def plot(index):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 10 * np.log10(np.maximum(np.square(np.abs(signal_stft)), 10 ** (-15)))
        im = ax.imshow(image_batch[index, 0], cmap='viridis',
                       origin='lower',
                       aspect='auto')
        fig.colorbar(im, orientation="vertical", pad=0.2)
        plt.show()

    cmap = cm.ScalarMappable(matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), 'viridis')

    rgba_data = cmap.to_rgba(image_batch, norm=True)
    rgb_data = np.moveaxis(np.squeeze(rgba_to_rgb(rgba_data)), 3, 1)
    image_data = torch.from_numpy(rgb_data)
    norm_image_data = ((image_data + 1) * 127.5).type(torch.ByteTensor)
    grid = make_grid(norm_image_data, nrow=n_img_per_row, padding=2, normalize=False)

    return grid
