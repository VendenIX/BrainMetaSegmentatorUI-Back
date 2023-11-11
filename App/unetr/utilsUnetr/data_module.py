"""Module that contains the definition of a PyTorch
Lightning data module for the meta dataset."""

from functools import partial
import os
import sys
from typing import Callable, Optional, Tuple

META_MODULE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(META_MODULE))

from monai import transforms
from monai.data import DataLoader
import pytorch_lightning as pl
import torch

from meta.data import DatasetType, MetaDatalist, MetaReader
from meta.data.cache_dataset import CacheMetaSubset
from .transforms import CropBedd, RandCropByPosNegLabeld, ResizeOrDoNothingd


class MetaDataModule(pl.LightningDataModule):
    """Class which wraps the dataset and datalist classes
    to manipulate them easily with PyTorch Lightning.
    
    Attributes:
        data_dir: Directory of the data.
        json_datalist_filename: Datalist JSON filename.
        reader_type: Type of reader to read dataset.
        generator_seed: Seed for the random generator to split the data.
        train_transform: Train data transform.
        val_transform: Validation/test/prediction data transform.
        train_batch_size: Batch size of the train and predict dataloaders.
        val_batch_size: Batch size of the validation and test dataloaders.
        workers: Number of workers to load data samples.
        use_cached_dataset: Activate the cache dataset.
    """
    def __init__(
        self,
        data_dir: str,
        json_datalist_filename: str,
        reader_type: MetaReader,
        use_cached_dataset: Optional[bool] = True,
        train_batch_size: Optional[int] = 1,
        val_batch_size: Optional[int] = 1,
        workers: Optional[int] = 1,
        generator_seed: Optional[int] = None,
        precision: Optional[int] = 32,
        voxel_space: Optional[Tuple[float, float, float]] = None,
        a_min: Optional[float] = None,
        a_max: Optional[float] = None,
        b_min: Optional[float] = None,
        b_max: Optional[float] = None,
        clip: Optional[bool] = False,
        crop_bed_max_number_of_rows_to_remove: Optional[int] = None,
        crop_bed_max_number_of_cols_to_remove: Optional[int] = None,
        crop_bed_min_spatial_size: Optional[Tuple[int, int, int]] = None,
        enable_fgbg2indices_feature: Optional[bool] = None,
        pos: Optional[float] = None,
        neg: Optional[float] = None,
        num_samples: Optional[int] = 1,
        roi_size: Optional[Tuple[int, int, int]] = None,
        random_flip_prob: Optional[float] = None,
        random_90_deg_rotation_prob: Optional[float] = None,
        random_intensity_scale_prob: Optional[float] = None,
        random_intensity_shift_prob: Optional[float] = None,
        val_resize: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Arguments:
            data_dir: Directory of the data.
            json_datalist_filename: Datalist JSON filename.
            reader_type: Type of reader to read dataset.
            use_cached_dataset: Activate the cache dataset.
            train_batch_size: Batch size of the train and predict dataloaders.
            val_batch_size: Batch size of the validation and test dataloaders.
            workers: Number of workers to load data samples.
            generator_seed: Seed for the random generator to split the data.
            precision: Tensor floating point precision.
            voxel_space: Output voxel spacing.
            a_min: Intensity original range min.
            a_max: Intensity original range max.
            b_min: Intensity target range min.
            b_max: Intensity target range max.
            clip: Clip the intensity if target values are not between `b_min` and `b_max`.
            crop_bed_max_number_of_rows_to_remove: Max number of rows to remove bed from the image.
            crop_bed_max_number_of_cols_to_remove: Max number of columns to remove bed from the image.
            crop_bed_min_spatial_size: Minimum spatial size to avoid to crop bodies. 
                Note that the third value is only indicative and if a value of -1 is passed, the dimension is next.
            enable_fgbg2indices_feature: Enable the instance of ``FgBgToIndicesd`` to determine the samples to extract from the label mask.
            pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
                to pick a foreground voxel as a center rather than a background voxel.
            neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
                to pick a foreground voxel as a center rather than a background voxel.
            num_samples: Number of samples to generate for data augmentation.
            roi_size: Size of the regions of interest to extract.
            random_flip_prob: Probability to randomly flip the image.
            random_90_deg_rotation_prob: Probability to randomly rotate by 90 degrees.
            random_intensity_scale_prob: Probability to randomly scale the intensity of the input image.
            random_intensity_shift_prob: Probability to randomly shift intensity of the input image.
            val_resize: Spatial size for the validation images (run at the beginning of the validation transform).
        """
        super().__init__()

        assert precision in (16, 32, 64)
        self.dtype = torch.float32
        if precision == 16:
            self.dtype = torch.float16
        elif precision == 64:
            self.dtype = torch.float64

        self.data_dir = data_dir
        self.json_file_datalist = os.path.join(data_dir, json_datalist_filename)
        self.reader_type = reader_type
        self.generator_seed = generator_seed
        
        self.train_transform, self.val_transform, self.test_transform = self.__get_transforms(
            self.dtype, voxel_space, a_min, a_max, b_min, b_max, clip, 
            crop_bed_max_number_of_rows_to_remove, crop_bed_max_number_of_cols_to_remove,
            crop_bed_min_spatial_size, enable_fgbg2indices_feature, pos, neg, 
            num_samples, roi_size, random_flip_prob, random_90_deg_rotation_prob, 
            random_intensity_scale_prob, random_intensity_shift_prob, val_resize
        )

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.workers = workers
        self.use_cached_dataset = use_cached_dataset

        self.save_hyperparameters()
        
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._predict_dataset = None
        
        
    
    def __create_datalist(self, data_dir: str, json_path: str, reader_type: MetaReader,
                          lengths: Optional[Tuple[float, float, float]] = None, deactivate_shuffle: bool = False) -> MetaDatalist:
        """Creates a datalist instance to easily manipulate the different datasets 
        inside the module.

        If the `json_path` exists, it directly loads the datalist from it.
        If you provide `lengths` and `deactivate_shuffle` is to `True`,
        it returns a datalist that have different not shuffled dataset.
        Else, it returns a new datalist with default spliting lengths,
        arguments and attributes passed through the calls.

        The two last cases save a JSON file accordingly to the `json_path`
        to easily retrieve specific wanted datalist.

        Arguments:
            data_dir: Base directory of the data.
            json_path: Path of the JSON file to save/load the created datalist to 
                have consistency in the training.
            reader_type: Type of reader to load the dataset files according to
                the files architecture used.
            lengths: Describes the way of spliting the global dataset on sub-dataset
                (for training, validation and testing). (Default to `None`).
            deactivate_shuffle: Deactivate the random shuffling when instanciating
                the datasets. (Default to `False`).
        
        Returns:
            An instanciated datalist corresponding to the passed arguments.
        
        See also:
            meta.data.datalist.MetaDatalist: The datalist class.
        """

        if os.path.exists(json_path):
            return MetaDatalist.from_json(json_path, data_dir, train_transform=self.train_transform,
                                              val_transform=self.val_transform, test_transform=self.test_transform, 
                                              dtype=self.dtype)
        
        datalist = None
        if lengths is not None and deactivate_shuffle:
            datalist = MetaDatalist(data_dir, reader_type, train_transform=self.train_transform,
                                    val_transform=self.val_transform, test_transform=self.test_transform,
                                    generator_seed=self.generator_seed, deactivate_shuffle=deactivate_shuffle, 
                                    lengths=lengths, dtype=self.dtype)
        else:
            datalist = MetaDatalist(data_dir, reader_type, train_transform=self.train_transform,
                                    val_transform=self.val_transform, test_transform=self.test_transform,
                                    generator_seed=self.generator_seed, deactivate_shuffle=deactivate_shuffle, 
                                    dtype=self.dtype)
        datalist.to_json(json_path)
        return datalist
    
    def setup(self, stage: str) -> None:
        """Setup a datalist according to the considered stage.
        
        Initializes a datalist. If we are in a predict stage,
        all the data go in test subset to predict all the data.
        Else, default passed values at class instanciation are
        used.

        Arguments:
            stage: What type of stage we are in. Can be `"predict"`, `"fit"`,
                `"validation"` or `"test"`.
        
        See also:
            __create_datalist
        """
        if stage == "predict":
            self.datalist = self.__create_datalist(self.data_dir, self.json_file_datalist, self.reader_type,
                                                   lengths=(0.0, 0.0, 1.0), deactivate_shuffle=True)
        else:
            self.datalist = self.__create_datalist(self.data_dir, self.json_file_datalist, self.reader_type)
    
    def __get_transforms(
            self,
            dtype = torch.float32,
            voxel_space: Optional[Tuple[float, float, float]] = None,
            a_min: Optional[float] = None,
            a_max: Optional[float] = None,
            b_min: Optional[float] = None,
            b_max: Optional[float] = None,
            clip: Optional[bool] = False,
            crop_bed_max_number_of_rows_to_remove: Optional[int] = None,
            crop_bed_max_number_of_cols_to_remove: Optional[int] = None,
            crop_bed_min_spatial_size: Optional[Tuple[int, int, int]] = None,
            enable_fgbg2indices_feature: Optional[bool] = None,
            pos: Optional[float] = None,
            neg: Optional[float] = None,
            num_samples: Optional[int] = 1,
            roi_size: Optional[Tuple[int, int, int]] = None,
            random_flip_prob: Optional[float] = None,
            random_90_deg_rotation_prob: Optional[float] = None,
            random_intensity_scale_prob: Optional[float] = None,
            random_intensity_shift_prob: Optional[float] = None,
            val_resize: Optional[Tuple[int, int, int]] = None,
        ) -> Tuple[transforms.Transform, transforms.Transform]:
        """Returns the data transform pipelines to preprocess the dataset.

        Some random transforms are put for the training transform to perform
        some data augmentation to have a more generalizable model.
        
        Arguments:
            dtype: Tensor floating point precision in PyTorch.
            voxel_space: Output voxel spacing.
            a_min: Intensity original range min.
            a_max: Intensity original range max.
            b_min: Intensity target range min.
            b_max: Intensity target range max.
            clip: Clip the intensity if target values are not between `b_min` and `b_max`.
            crop_bed_max_number_of_rows_to_remove: Max number of rows to remove bed from the image.
            crop_bed_max_number_of_cols_to_remove: Max number of columns to remove bed from the image.
            crop_bed_min_spatial_size: Minimum spatial size to avoid to crop bodies. 
                Note that the third value is only indicative and if a value of -1 is passed, the dimension is next.
            enable_fgbg2indices_feature: Enable the instance of ``FgBgToIndicesd`` to determine the samples to extract from the label mask.
            pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
                to pick a foreground voxel as a center rather than a background voxel.
            neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
                to pick a foreground voxel as a center rather than a background voxel.
            num_samples: Number of samples to generate for data augmentation.
            roi_size: Size of the regions of interest to extract.
            random_flip_prob: Probability to randomly flip the image.
            random_90_deg_rotation_prob: Probability to randomly rotate by 90 degrees.
            random_intensity_scale_prob: Probability to randomly scale the intensity of the input image.
            random_intensity_shift_prob: Probability to randomly shift intensity of the input image.
            val_resize: Spatial size for the validation images (run at the beginning of the validation transform).

        Returns:
            transforms: Train and validation transform pipelines.
        
        See also:
            transforms.Orientation, transforms.Spacing, transforms.ScaleIntensityRange,
            transforms.CropForeground, transforms.RandSpatialCropSamples, transforms.ResizeWithPadOrCrop,
            transforms.RandFlip, transforms.RandRotate90, transforms.RandScaleIntensity,
            transforms.RandShiftIntensity, transforms.AsChannelFirst, transforms.ToTensor
        """

        spacing = transforms.Identity()
        if all([space > 0.0 for space in voxel_space]):
            spacing = transforms.Spacingd(
                keys=["image", "label"], pixdim=voxel_space, mode=("bilinear", "nearest")
            ) # to change the dimension of the voxel to have less data to compute
        
        posneg_label_croper_kwargs = {
            "keys": ["image", "label"],
            "label_key": "label",
            "spatial_size": roi_size,
            "pos": pos,
            "neg": neg,
            "num_samples": num_samples,
            "image_key": "image",
            "allow_smaller": True,
        }
        fgbg2indices = transforms.Identity()
        if enable_fgbg2indices_feature:
            fgbg2indices = transforms.FgBgToIndicesd(
                keys=["image", "label"], image_key="label", image_threshold=0.0
            ) # to crop samples close to the label mask
            posneg_label_croper_kwargs["fg_indices_key"] = "image_fg_indices"
            posneg_label_croper_kwargs["bg_indices_key"] = "image_bg_indices"
        else:
            posneg_label_croper_kwargs["image_threshold"] = 0.0

        train_transform = transforms.Compose(
            [
                transforms.Orientationd(keys=["image", "label"], axcodes="LAS"), # to have the same orientation
                spacing,
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip
                ), # scales image from a values to b values
                CropBedd(
                    keys=["image", "label"], image_key="image",
                    max_number_of_rows_to_remove=crop_bed_max_number_of_rows_to_remove,
                    max_number_of_cols_to_remove=crop_bed_max_number_of_cols_to_remove,
                    min_spatial_size=crop_bed_min_spatial_size,
                    axcodes_orientation="LAS",
                ), # crop the bed from the image (useless data)
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"), # remove useless background image part
                fgbg2indices,
                RandCropByPosNegLabeld(**posneg_label_croper_kwargs), # extract some random samples from the current image
                transforms.ResizeWithPadOrCropd(
                    keys=["image", "label"], spatial_size=roi_size,
                ), # pad image if the previous crop has not the correct ROI size
                transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=0), # random flip on the X axis
                transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=1), # random flip on the Y axis
                transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=2), # random flip on the Z axis
                transforms.RandRotate90d(keys=["image", "label"], prob=random_90_deg_rotation_prob, max_k=3), # random 90 degree rotation
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=random_intensity_scale_prob), # random intensity scale
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=random_intensity_shift_prob), # random intensity shifting
                transforms.ToTensord(keys=["image", "label"], dtype=dtype), # to have a PyTorch tensor as output
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Orientationd(keys=["image", "label"], axcodes="LAS"), # to have the same orientation
                spacing,
                ResizeOrDoNothingd(keys=["image", "label"], max_spatial_size=val_resize, cut_slices=True, axcodes_orientation="LAS"),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip
                ), # scales image from a values to b values
                CropBedd(
                    keys=["image", "label"], image_key="image",
                    max_number_of_rows_to_remove=crop_bed_max_number_of_rows_to_remove,
                    max_number_of_cols_to_remove=crop_bed_max_number_of_cols_to_remove,
                    min_spatial_size=crop_bed_min_spatial_size,
                    axcodes_orientation="LAS",
                ), # crop the bed from the image (useless data)
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"), # remove useless background image part
                transforms.ToTensord(keys=["image", "label"], dtype=dtype), # to have a PyTorch tensor as output
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Orientationd(keys=["image"], axcodes="LAS"), # to have the same orientation
                spacing,
                ResizeOrDoNothingd(keys=["image"], max_spatial_size=val_resize, cut_slices=True, axcodes_orientation="LAS"),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip
                ), # scales image from a values to b values
                CropBedd(
                    keys=["image"], image_key="image",
                    max_number_of_rows_to_remove=crop_bed_max_number_of_rows_to_remove,
                    max_number_of_cols_to_remove=crop_bed_max_number_of_cols_to_remove,
                    min_spatial_size=crop_bed_min_spatial_size,
                    axcodes_orientation="LAS",
                ), # crop the bed from the image (useless data)
                transforms.CropForegroundd(keys=["image"], source_key="image"), # remove useless background image part
                transforms.ToTensord(keys=["image"], dtype=dtype), # to have a PyTorch tensor as output
            ]
        )
        
        return train_transform, val_transform, test_transform

    def _get_partial_dataloader(self, batch_size: int, shuffle: bool) -> Callable:
        """Returns a partial dataloader initializer. Only shared
        arguments are passed here.
        
        Arguments:
            batch_size: Number of samples to pass for each batch.
            shuffle: Activation of the random shuffling of data.
        
        Returns:
            dataloader_class: Partial pre-instanciated class.
        
        See also:
            partial, Dataloader
        """
        return partial(DataLoader, batch_size=batch_size, shuffle=shuffle, num_workers=self.workers, pin_memory=False, persistent_workers=False)
    
    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader.
        
        If a cached dataset need to be used, the class stores
        an instanciated dataloader and returns it. If it already
        created, it only returns it.
        Else, the associated dataloader is returned with the
        corresponding data subset.
        
        Returns:
            dataloader: Instanciated dataloader.
        
        See also:
            _get_partial_dataloader: Partial dataloader class.
            meta.data.cache_dataset.CacheMetaSubset.from_meta_subset: Used when the cached dataset is used.
        """
        if self.use_cached_dataset and self._train_dataset is not None:
            return self._train_dataset
        
        partial_dataloader = self._get_partial_dataloader(self.train_batch_size, True)
        
        # store the cache dataset
        if self.use_cached_dataset:
            self._train_dataset = partial_dataloader(
                CacheMetaSubset.from_meta_subset(self.datalist.get_subset(DatasetType.TRAINING), self.workers)
            )
            return self._train_dataset

        return partial_dataloader(self.datalist.get_subset(DatasetType.TRAINING))
    
    def val_dataloader(self) -> DataLoader:
        """Returns the validation dataloader.
        
        If a cached dataset need to be used, the class stores
        an instanciated dataloader and returns it. If it already
        created, it only returns it.
        Else, the associated dataloader is returned with the
        corresponding data subset.
        
        Returns:
            dataloader: Instanciated dataloader.
        
        See also:
            _get_partial_dataloader: Partial dataloader class.
            meta.data.cache_dataset.CacheMetaSubset.from_meta_subset: Used when the cached dataset is used.
        """
        if self.use_cached_dataset and self._val_dataset is not None:
            return self._val_dataset
        
        partial_dataloader = self._get_partial_dataloader(self.val_batch_size, False)
        
        # store the cache dataset
        if self.use_cached_dataset:
            self._val_dataset = partial_dataloader(
                CacheMetaSubset.from_meta_subset(self.datalist.get_subset(DatasetType.VALIDATION), self.workers),
            )
            return self._val_dataset
        
        return partial_dataloader(self.datalist.get_subset(DatasetType.VALIDATION))
    
    def test_dataloader(self) -> DataLoader:
        """Returns the test dataloader.
        
        If a cached dataset need to be used, the class stores
        an instanciated dataloader and returns it. If it already
        created, it only returns it.
        Else, the associated dataloader is returned with the
        corresponding data subset.
        
        Returns:
            dataloader: Instanciated dataloader.
        
        See also:
            _get_partial_dataloader: Partial dataloader class.
            meta.data.cache_dataset.CacheMetaSubset.from_meta_subset: Used when the cached dataset is used.
        """
        if self.use_cached_dataset and self._test_dataset is not None:
            return self._test_dataset
        
        partial_dataloader = self._get_partial_dataloader(self.val_batch_size, False)

        # store the cache dataset
        if self.use_cached_dataset:
            self._test_dataset = partial_dataloader(
                CacheMetaSubset.from_meta_subset(self.datalist.get_subset(DatasetType.TESTING), self.workers),
                batch_size=self.val_batch_size,
            )
            return self._test_dataset
        
        return partial_dataloader(self.datalist.get_subset(DatasetType.TESTING))
    
    def predict_dataloader(self) -> DataLoader:
        """Returns the predict dataloader.
        
        If a cached dataset need to be used, the class stores
        an instanciated dataloader and returns it. If it already
        created, it only returns it.
        Else, the associated dataloader is returned with the
        corresponding data subset.
        
        Returns:
            dataloader: Instanciated dataloader.
        
        See also:
            _get_partial_dataloader: Partial dataloader class.
            meta.data.cache_dataset.CacheMetaSubset.from_meta_subset: Used when the cached dataset is used.
        """
        if self.use_cached_dataset and self._predict_dataset is not None:
            return self._predict_dataset
        
        partial_dataloader = self._get_partial_dataloader(self.val_batch_size, False)

        # store the cache dataset
        if self.use_cached_dataset:
            self._predict_dataset = partial_dataloader(
                CacheMetaSubset.from_meta_subset(self.datalist.get_subset(DatasetType.TESTING), self.workers),
            )
            return self._predict_dataset
        
        return partial_dataloader(self.datalist.get_subset(DatasetType.TESTING))
