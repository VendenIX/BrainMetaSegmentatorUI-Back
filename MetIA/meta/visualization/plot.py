from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from .core import (
    array_preprocessing_numpy,
    create_masked_image,
)


def plot_image(
    img: Union[np.ndarray, torch.Tensor],
    title: Optional[str] = None,
    cmap: Union[Tuple[int, int, int], str] = "bone",
    figsize: Tuple[int, int] = (9, 9)
) -> None:
    """show the image
    apply the preprocessing to a numpy array
    create the figure with the wanted size
    determine the image that we want to show and is color
    dont put label on axis
    define the title
    show the figure
    Args:
    img: torche tensor ou ndarray numpy
    title: string avec non pour la figure
    cmap: string or tuple specify the color map
    figsize: wanted size for the figure"""
    img = array_preprocessing_numpy(img)
    plt.figure(figsize=figsize)

    plt.imshow(img, cmap=cmap)
    plt.axis("off")

    plt.title(title)
    plt.show()


def plot_image_and_mask(
    img: np.ndarray,
    seg_mask: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 9)
) -> None:
    """show the image and its mask
    create the figure with the wanted size for tow image
    determine the image that we want to show and is color
    dont put label on axis
    determine the second image which is the mask
    dont put label on axis
    define the title
    show the figure
    Args:
    img: torche tensor ou ndarray numpy
    seg_mask: torche tensor ou ndarray numpy
    title: string avec non pour la figure
    figsize: wanted size for the figure"""
    fig, axis = plt.subplots(1, 2, figsize=figsize)

    axis[0].imshow(img, cmap="bone")
    axis[0].axis("off")

    axis[1].imshow(seg_mask)
    axis[1].axis("off")

    plt.title(title)
    plt.show()


def plot_masked_image(
    img: np.ndarray,
    seg_mask: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 9),
    alpha: float = 0.5,
    color: np.ndarray = np.array([255, 0, 0])
) -> None:
    """show the image
    apply the preprocessing to a numpy array on img ans seg_mask
    combine the tow images to one with the color and alpha wanted for the label
    create the figure with the wanted size
    determine the image that we want to show so the combine one
    dont put label on axis
    define the title
    show the figure
    Args:
    img: torche tensor ou ndarray numpy
    seg_mask:torche tensor ou ndarray numpy
    title: string avec non pour la figure
    color: tuple that define the color for the mask
    alpha: determine the transparence of the mask put in the image
    figsize: wanted size for the figure"""
    img = array_preprocessing_numpy(img)
    seg_mask = array_preprocessing_numpy(seg_mask)

    image_combined = create_masked_image(img, seg_mask, alpha=alpha, color=color)

    fig = plt.figure(figsize=figsize)
    plt.imshow(image_combined.transpose([1, 2, 0]))
    plt.axis("off")
    plt.title(title)
    plt.show()


def plot_3d_mask(
    seg_mask: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 9),
    color: np.ndarray = np.array([255, 0, 0])
) -> None:
    """show the mask as a 3D volume
    apply the preprocessing to a numpy array on seg_mask
    create the figure with the wanted size and get the 3D axis
    add mask as voxels with the passed color
    define the title
    show the figure
    Args:
    seg_mask:torche tensor ou ndarray numpy
    title: string avec non pour la figure
    color: tuple that define the color for the mask
    figsize: wanted size for the figure"""
    seg_mask = array_preprocessing_numpy(seg_mask).transpose([1, 2, 0])

    ax = plt.figure(figsize=figsize).add_subplot(projection="3d")
    ax.voxels(seg_mask, facecolors=color, edgecolor=color)
    plt.axis("off")
    plt.title(title)
    plt.show()
