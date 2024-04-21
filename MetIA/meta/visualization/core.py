from typing import TypeVar, Union

import cv2 as cv
import numpy as np
import torch

from meta.arrays import all2np, all2torch


Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


def array_preprocessing_numpy(
    array: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """preprocesse an array which is a tensor or a ndarray 
    convert the array to a numpy array
    convert the type of the array to float
    get the min value of array
    get the max value of array
    scale the array, min value=0 and max=1
    pass the max value to 255 and the min to 0
    Args:
    array: ndarray or tensor
    Returns: the array"""
    array = all2np(array)
    array = array.astype(float)

    min_ = array.min()
    max_ = array.max()

    array = (array - min_)/(max_ - min_ + 1e-9)
    array = (array*255).astype(np.uint8)

    return array


def array_preprocessing_torch(
    array: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """preprocesse an array which is a tensor or a ndarray 
    convert the array to a torch tensor
    convert the type of the array to float
    get the min value of array
    get the max value of array
    scale the array, min value=0 and max=1
    pass the max value to 255 and the min to 0
    Args:
    array: ndarray or tensor
    Returns: the tensor"""
    array = all2torch(array)
    array = array.to(float)

    min_ = array.min()
    max_ = array.max()

    array = (array - min_)/(max_ - min_ + 1e-9)
    array = (array*255).to(torch.uint8)

    return array


def permute_axis_for_visualization(
    img: Tensor,
    mask: Tensor,
) -> Tensor:
    """permute the axis of a torch tensor
    verify if img is a numpy array
    if it is:
        scales it to 255 and 0 with the function preprocessing for numpy
        transpose img
    else:
        scales it with the function for torch tensor
        permute img
    we apply the same to mask
    Args:
    img: torch tensor
    mask:torch tensor
    Returns: the image and its mask"""
    if isinstance(img, np.ndarray):
        img = array_preprocessing_numpy(img)
        img = img.transpose(0, 3, 1, 2)[0]
    else:
        img = array_preprocessing_torch(img)
        img = img.permute(0, 3, 1, 2)[0]

    if isinstance(mask, np.ndarray):
        mask = array_preprocessing_numpy(mask)
        mask = mask.transpose(0, 3, 1, 2)[0]
    else:
        mask = array_preprocessing_torch(mask)
        mask = mask.permute(0, 3, 1, 2)[0]

    return img, mask


def create_masked_image(
    img: np.ndarray,
    seg_mask: np.ndarray,
    alpha: float = 0.5,
    color: np.ndarray = np.array([255, 0, 0])
) -> np.ndarray:
    """create an image with in base img and on it a mask in transparence
    reshape the color
    expand the dimension of the mask to 3D
    same for img
    create a mask array which separated a boolean and the image
    fill the value of the mask precedently create
    conbine the image with a transparence
    Args:
    img: ndarray
    seg_mask:nd array
    alpha: determine the transparence of the mask in the image
    color: the color of the mask put in the image
    Returns: the combined image"""

    color = np.asarray(color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(seg_mask, 0).repeat(3, axis=0)
    img = np.expand_dims(img, 0).repeat(3, axis=0)

    masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    image_combined = cv.addWeighted(img, 1 - alpha, image_overlay, alpha, 0)
    return image_combined/np.max(image_combined)
