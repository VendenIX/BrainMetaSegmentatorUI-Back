import asyncio
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .core import create_masked_image


def make_overlay_gif_animation(
    images: np.ndarray,
    seg_masks: np.ndarray,
    file_output: str = "anim.gif",
    figsize: Tuple[int, int] = (9, 9),
    alpha: float = 0.5,
    seg_color: np.ndarray = np.array([255, 0, 0]),
    keep_intermediate_files: bool = False
) -> None:
    """make a giff animation
        get a loop event
        create a loop with the images create by the function create_anim_image 
        create files with the images and open all the images in them
        save the gif
        close all the file created before
        if the argument keep_intermediate_files is in false:
            remove alle the closed files
        Args:
        images: an array with all the images we need
        seg_masks: the labels of the images
        file_output: name of the saved file
        figsize: size of the gif
        alpha: transparence canal
        seg_color: color of labels put in images
        keep_intermediate_files: boolean which say if we keep the created files"""
    intermediate_files = file_output.split(".")[0] + "{0}.png"

    async def create_anim_image(i: int) -> None:
        """create the image for the gif
        create a figure
        get the image and it mask
        combine the mask and the image with a transparence of alpha and the wanted seg_color for the mask

        show the transposed image
        save the fig
        Args:
        i: the identifiant of the wanted image"""
        fig = plt.figure(figsize=figsize)
        image = images[i]
        mask = seg_masks[i,:,:]

        image_combined = create_masked_image(image, mask, alpha=alpha, color=seg_color)

        plt.imshow(image_combined.transpose([1, 2, 0]), cmap='gray')
        plt.axis("off")
        #plt.show()
        plt.savefig(intermediate_files.format(i))
        plt.close()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*[
        create_anim_image(i) for i in range(images.shape[0])
    ]))
    
    files = [intermediate_files.format(i) for i in range(images.shape[0])]
    imgs = [Image.open(f) for f in files]

    imgs[0].save(fp=file_output, format="GIF", append_images=imgs[1:], save_all=True, loop=0, duration=images.shape[0] // 2)
    for im in imgs:
        im.close()

    # remove useless files
    if not keep_intermediate_files and os.path.exists(file_output):
        for f in files:
            os.remove(f)
