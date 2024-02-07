"""Module that contains the definition of different
transforms for data pipeline.

The large majority of code came from this file https://github.com/Project-MONAI/MONAI/blob/0.8.0/monai/transforms/croppad/dictionary.py
et from this one https://github.com/Project-MONAI/MONAI/blob/0.8.0/monai/transforms/utils.py.
"""

from copy import deepcopy
from itertools import chain
import math
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

from monai import transforms
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad.array import (
    BorderPad,
    RandCropByPosNegLabel,
    SpatialCrop,
)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.dictionary import InterpolateModeSequence
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utils import map_binary_to_indices
from monai.transforms.utils_pytorch_numpy_unification import unravel_index
from monai.utils import (
    ImageMetaKey as Key,
    InterpolateMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
from monai.utils.enums import InverseKeys
from monai.utils.type_conversion import convert_data_type
import numpy as np
import torch


class SampleNormalizer():
    """Class that allows to normalize data
    between 0 and 1 included."""

    def __call__(self, data: torch.Tensor, dtype=torch.uint8) -> torch.Tensor:
        """Computes the "Min-Max Normalization" of the `data` tensor.
        
        Arguments:
            data: Input tensor to normalize.
            dtype: Type to convert the tensor after normalization.
            
        Returns:
            data: Normalized and converted input tensor.
        """

        # simplify normalization if boolean values
        data = data.to(float)
        
        min_ = data.min()
        max_ = data.max()

        # avoid useless computations
        if min_ == max_:
            if min_ != 0: # return a tensor of 1
                return data / min_
            
            # avoid 0 division error
            return data
        
        # min-max normalization
        return ((data - min_) / (max_ - min_)).to(dtype)


class CropBedd(transforms.MapTransform):
    """
    Dictionary-based class.
    It crops the bed from the passed images.
    """
    def __init__(self, keys: KeysCollection, image_key: str = "image",
                 max_number_of_rows_to_remove: int = 90, 
                 max_number_of_cols_to_remove: int = 90, 
                 axcodes_orientation: str = "LAS",
                 min_spatial_size: Tuple[int, int, int] = None,
                 allow_missing_keys: bool = False) -> None:
        """
        Arguments:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            max_number_of_rows_to_remove: Max number of rows to remove in the image.
            max_number_of_cols_to_remove: Max number of columns to remove in the image.
            axcodes_orientation: Orientation of the image.
            min_spatial_size: Minimum spatial size to avoid to crop bodies.
                Note that the third value is only indicative and if a value of -1 is passed, the dimension is next.
            allow_missing_keys: don't raise exception if key is missing.
        
        See also:
            nibabel.orientations.ornt2axcodes
        """
        super().__init__(keys, allow_missing_keys)
        self.image_key = image_key
        self.max_number_of_rows_to_remove = max(0, max_number_of_rows_to_remove)
        self.max_number_of_cols_to_remove = max(0, max_number_of_cols_to_remove)
        self.min_spatial_size = min_spatial_size
        self.axcodes_orientation = axcodes_orientation
    
    def __call__(self, data):
        # nothing to remove
        if self.max_number_of_cols_to_remove == 0 and self.max_number_of_rows_to_remove == 0:
            return data
        
        img_size = data[self.image_key].shape

        max_number_of_rows_to_remove = 0
        if img_size[1] > self.min_spatial_size[0]:
            max_number_of_rows_to_remove = min(img_size[1] - self.min_spatial_size[0], self.max_number_of_rows_to_remove)
        max_number_of_cols_to_remove = 0
        if img_size[2] > self.min_spatial_size[1]:
            max_number_of_cols_to_remove = min(img_size[2] - self.min_spatial_size[1], self.max_number_of_cols_to_remove)

        roi_size = (
            img_size[1] - max_number_of_rows_to_remove, # remove some rows
            img_size[2] - max_number_of_cols_to_remove, # remove some columns
            img_size[3] # we don't want to remove slices
        )
        roi_center = (
            math.ceil(roi_size[0] / 2) + int(self.axcodes_orientation[0] == "R") * max_number_of_rows_to_remove, # move the X axis center because we want the body (with the R** orientation, the body is in the bottom)
            math.ceil((roi_size[1] + max_number_of_cols_to_remove) / 2), # centered in Y axis because we want the body in the center of this axis 
            math.ceil(roi_size[2] / 2) # just get the center in the Z axis
        )

        cropper = transforms.SpatialCropd(
            keys=self.keys,
            roi_center=roi_center,
            roi_size=roi_size,
            allow_missing_keys=self.allow_missing_keys
        )

        return cropper(data)


def correct_crop_centers(
    centers: List[Union[int, torch.Tensor]],
    spatial_size: Union[Sequence[int], int],
    label_spatial_shape: Sequence[int],
    allow_smaller: bool = False,
):
    """
    Utility to correct the crop center if the crop size and centers are not compatible with the image size.
    Args:
        centers: pre-computed crop centers of every dim, will correct based on the valid region.
        spatial_size: spatial size of the ROIs to be sampled.
        label_spatial_shape: spatial shape of the original label data to compare with ROI.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if any(np.subtract(label_spatial_shape, spatial_size) < 0):
        if not allow_smaller:
            raise ValueError("The size of the proposed random crop ROI is larger than the image size.")
        spatial_size = tuple(min(l, s) for l, s in zip(label_spatial_shape, spatial_size))

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i, valid_s in enumerate(valid_start):
        # need this because np.random.randint does not work with same start and end
        if valid_s == valid_end[i]:
            valid_end[i] += 1
    valid_centers = []
    for c, v_s, v_e in zip(centers, valid_start, valid_end):
        _c = int(convert_data_type(c, np.ndarray)[0])  # type: ignore
        center_i = min(max(_c, v_s), v_e - 1)
        valid_centers.append(int(center_i))
    return valid_centers


def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: NdarrayOrTensor,
    bg_indices: NdarrayOrTensor,
    rand_state: Optional[np.random.RandomState] = None,
    allow_smaller: bool = False,
) -> List[List[int]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]
    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.
    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = unravel_index(idx, label_spatial_shape)
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return centers


class RandCropByPosNegLabeld(Randomizable, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding meta data.
    And will return a list of dictionaries for all the cropped images.
    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.
    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the meta data is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to to fetch the meta data according
            to the key data, default is `meta_dict`, the meta data is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        allow_missing_keys: don't raise exception if key is missing.
    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.
    """

    backend = RandCropByPosNegLabel.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: Optional[List[List[int]]] = None
        self.allow_smaller = allow_smaller

    def randomize(
        self,
        label: NdarrayOrTensor,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = d.pop(self.fg_indices_key, None) if self.fg_indices_key is not None else None
        bg_indices = d.pop(self.bg_indices_key, None) if self.bg_indices_key is not None else None

        self.randomize(label, fg_indices, bg_indices, image)
        if not isinstance(self.spatial_size, tuple):
            raise ValueError("spatial_size must be a valid tuple.")
        if self.centers is None:
            raise ValueError("no available ROI centers to crop.")

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [dict(d) for _ in range(self.num_samples)]

        for i, center in enumerate(self.centers):
            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])
            for key in self.key_iterator(d):
                img = d[key]
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
                orig_size = img.shape[1:]
                results[i][key] = cropper(img)
                self.push_transform(results[i], key, extra_info={"center": center}, orig_size=orig_size)
            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[InverseKeys.ORIG_SIZE])
            current_size = np.asarray(d[key].shape[1:])
            center = transform[InverseKeys.EXTRA_INFO]["center"]
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
            # get required pad to start and end
            pad_to_start = np.array([s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)])
            pad_to_end = orig_size - current_size - pad_to_start
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class ResizeOrDoNothingd(transforms.MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Resize`.
    It resize the image only if the shape is greater than maximum expected.
    """

    backend = transforms.Resized.backend

    def __init__(
        self, 
        keys: KeysCollection,
        max_spatial_size: Union[Sequence[int], int],
        size_mode: str = "all",
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        cut_slices: bool = False,
        axcodes_orientation: str = "RAS",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Arguments:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            max_spatial_size: maximum expected shape and in case of the spatial dimensions are greater
                than this value, it become the spatial dimensions after resize operation.
                if some components of the `max_spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `max_spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            size_mode: should be "all" or "longest", if "all", will use `max_spatial_size` for all the spatial dims,
                if "longest", rescale the image so that only the longest side is equal to specified `max_spatial_size`,
                which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
                https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
                #albumentations.augmentations.geometric.resize.LongestMaxSize.
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``"area"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
                It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
            cut_slices: Removing instead of resizing the last dimension of the image.
            axcodes_orientation: Orientation of the image.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        
        self.max_spatial_size = max_spatial_size
        self.resize = transforms.Resized(
            keys=keys,
            spatial_size=max_spatial_size,
            size_mode=size_mode,
            mode=mode,
            align_corners=align_corners,
            allow_missing_keys=allow_missing_keys
        )
        self.cut_slices = cut_slices
        self.axcodes_orientation = axcodes_orientation
    
    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        if self.max_spatial_size == (-1, -1, -1):
            data["has_not_been_resized"] = True
            return data

        # shape verification
        have_all_same_correct_shape = True
        shape = None
        for key in self.keys:
            # same shape
            if shape is None:
                shape = data[key].shape
            else:
                have_all_same_correct_shape &= (data[key].shape == shape)
            
            # spatial size if less than max_spatial_size
            if isinstance(self.max_spatial_size, int):
                for ii in range(1, len(shape)):
                    have_all_same_correct_shape &= shape[ii] <= self.max_spatial_size
            else:
                for ii in range(len(self.max_spatial_size)):
                    if self.max_spatial_size[ii-1] != -1:
                        have_all_same_correct_shape &= shape[ii-1] <= self.max_spatial_size[ii-1]

        data["has_not_been_resized"] = have_all_same_correct_shape
        if have_all_same_correct_shape:
            return data
        
        if self.cut_slices and self.max_spatial_size[2] > 0 and data[self.keys[0]].shape[3] > self.max_spatial_size[2]:
            for key in self.keys:
                if self.axcodes_orientation[2] == "S":
                    data[key] = data[key][:,:,:,:self.max_spatial_size[2]]
                else:
                    data[key] = data[key][:,:,:,data[key].shape[3] - self.max_spatial_size[2]:]
        
        return self.resize(data)
    
    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        """
        Inverse of ``__call__``.

        If the data has been resized, the inverse method of the
        Resize class is called.
        """
        has_not_been_resized = data.pop("has_not_been_resized", False)

        if has_not_been_resized:
            return data
        
        return self.resize.inverse(data)
