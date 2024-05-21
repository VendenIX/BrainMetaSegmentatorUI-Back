"""Module that contains the fine-tuning CLI to test pipeline.

Created with the Pytorch LightningCLI, it allows
to easily choose any hyperparameters and
to quickly manage the stage type.
"""

import glob
import os
import sys
from typing import Optional, Tuple

META_MODULE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(META_MODULE)

from monai import transforms
from monai.apps import download_and_extract
from monai.data import DataLoader, Dataset, CacheDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning, EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

from meta.data.type_definition import MetaReader
from model_module import SegmentationTask
from unetr.utilsUnetr.data_module import MetaDataModule


class MSDDataModule(MetaDataModule):
    def __init__(self, data_dir: str, json_datalist_filename: str, reader_type: MetaReader, use_cached_dataset: Optional[bool] = True, train_batch_size: Optional[int] = 1, val_batch_size: Optional[int] = 1, workers: Optional[int] = 1, generator_seed: Optional[int] = None, precision: Optional[int] = 32, voxel_space: Optional[Tuple[float, float, float]] = None, a_min: Optional[float] = None, a_max: Optional[float] = None, b_min: Optional[float] = None, b_max: Optional[float] = None, clip: Optional[bool] = False, crop_bed_max_number_of_rows_to_remove: Optional[int] = None, crop_bed_max_number_of_cols_to_remove: Optional[int] = None, crop_bed_min_spatial_size: Optional[Tuple[int, int, int]] = None, enable_fgbg2indices_feature: Optional[bool] = None, pos: Optional[float] = None, neg: Optional[float] = None, num_samples: Optional[int] = 1, roi_size: Optional[Tuple[int, int, int]] = None, random_flip_prob: Optional[float] = None, random_90_deg_rotation_prob: Optional[float] = None, random_intensity_scale_prob: Optional[float] = None, random_intensity_shift_prob: Optional[float] = None, val_resize: Optional[Tuple[int, int, int]] = None) -> None:
        super().__init__(data_dir, json_datalist_filename, reader_type, use_cached_dataset, train_batch_size, val_batch_size, workers, generator_seed, precision, voxel_space, a_min, a_max, b_min, b_max, clip, crop_bed_max_number_of_rows_to_remove, crop_bed_max_number_of_cols_to_remove, crop_bed_min_spatial_size, enable_fgbg2indices_feature, pos, neg, num_samples, roi_size, random_flip_prob, random_90_deg_rotation_prob, random_intensity_scale_prob, random_intensity_shift_prob, val_resize)
        self.train_transform = self.__update_transforms(self.train_transform)
        self.val_transform = self.__update_transforms(self.val_transform)
        self.test_transform = self.__update_transforms(self.test_transform)

    def setup(self, stage: str) -> None:
        resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
        md5 = "410d4a301da4e5b2f6f86ec3ddba524e"
        ext = "*.nii.gz"

        os.makedirs(self.data_dir, exist_ok=True)

        extract_dir = os.path.join(self.data_dir, "Task09_Spleen")
        if not os.path.exists(extract_dir):
            compressed_file = os.path.join(self.data_dir, "Task09_Spleen.tar")
            download_and_extract(resource, compressed_file, self.data_dir, md5)
        
        train_images = sorted(
            glob.glob(os.path.join(extract_dir, "imagesTr", ext)))
        train_labels = sorted(
            glob.glob(os.path.join(extract_dir, "labelsTr", ext)))
        data_dicts = [
            {"image": image_name, "label": label_name, "patient_id": idx, "has_meta": False}
            for idx, (image_name, label_name) in enumerate(zip(train_images, train_labels))
        ]

        self.train_files, self.val_files = data_dicts[:-9], data_dicts[-9:]


        test_images = sorted(
            glob.glob(os.path.join(extract_dir, "imagesTs", ext)))
        self.test_files = [
            {"image": image_name, "label": None, "patient_id": idx, "has_meta": False}
            for idx, image_name in enumerate(test_images)
        ]
    
    def __update_transforms(self, transform: transforms.Transform) -> transforms.Transform:
        transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transform,
        ])
        return transform

    def train_dataloader(self) -> DataLoader:
        if self.use_cached_dataset and self._train_dataset is not None:
            return self._train_dataset
        
        partial_dataloader = self._get_partial_dataloader(self.train_batch_size, True)
        
        # store the cache dataset
        if self.use_cached_dataset:
            self._train_dataset = partial_dataloader(
                CacheDataset(self.train_files, transform=self.train_transform, num_workers=self.workers)
            )
            return self._train_dataset

        return partial_dataloader(Dataset(self.train_files, transform=self.train_transform))
    
    
    def val_dataloader(self) -> DataLoader:
        if self.use_cached_dataset and self._val_dataset is not None:
            return self._val_dataset
        
        partial_dataloader = self._get_partial_dataloader(self.train_batch_size, True)
        
        # store the cache dataset
        if self.use_cached_dataset:
            self._val_dataset = partial_dataloader(
                CacheDataset(self.val_files, transform=self.val_transform, num_workers=self.workers)
            )
            return self._val_dataset

        return partial_dataloader(Dataset(self.val_files, transform=self.val_transform))
    
    def test_dataloader(self) -> DataLoader:
        if self.use_cached_dataset and self._test_dataset is not None:
            return self._test_dataset
        
        partial_dataloader = self._get_partial_dataloader(self.train_batch_size, True)
        
        # store the cache dataset
        if self.use_cached_dataset:
            self._test_dataset = partial_dataloader(
                CacheDataset(self.train_files + self.val_files, transform=self.val_transform, num_workers=self.workers)
            )
            return self._test_dataset

        return partial_dataloader(Dataset(self.train_files + self.val_files, transform=self.val_transform))
    
    def predict_dataloader(self) -> DataLoader:
        if self.use_cached_dataset and self._predict_dataset is not None:
            return self._predict_dataset
        
        partial_dataloader = self._get_partial_dataloader(self.train_batch_size, True)
        
        # store the cache dataset
        if self.use_cached_dataset:
            self._predict_dataset = partial_dataloader(
                CacheDataset(self.test_files, transform=self.test_transform, num_workers=self.workers)
            )
            return self._predict_dataset

        return partial_dataloader(Dataset(self.test_files, transform=self.test_transform))


class UNETRFinetuningCLI(LightningCLI):
    """CLI to fine-tune a UNETR pretrained model."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments to a given parser.

        It adds to manage the backbone finetuning, the early stopping,
        and model checkpointing callbacks, link some same CLI arguments and add some
        directory specific CLI arguments.
        
        Arguments:
            parser: Parser which it adds some additional arguments.
        """

        # add default callbacks
        parser.add_lightning_class_args(BackboneFinetuning, "backbone_finetuning")
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({
            "early_stopping.monitor": "val_loss",
            "early_stopping.mode": "min",
            "early_stopping.patience": 5,
        })
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults({
            "model_checkpoint.monitor": "val_loss",
            "model_checkpoint.mode": "min",
            "model_checkpoint.filename": "checkpoint-{epoch:04d}-{val_loss:.3f}",
            "model_checkpoint.save_top_k": 1,
        })

        # links same arguments
        parser.link_arguments("data.precision", "trainer.precision")
        parser.link_arguments("model.max_epochs", "trainer.max_epochs")
        parser.link_arguments("data.roi_size", "model.roi_size")
        parser.link_arguments("model_checkpoint.monitor", "early_stopping.monitor")
        parser.link_arguments("model_checkpoint.mode", "early_stopping.mode")
        
        # add new arguments
        parser.add_argument("--checkpoint_dir_name", type=str, help="directory name to save checkpoints")
        parser.add_argument("--log_dir_name", type=str, help="directory name to save logs")
        parser.add_argument("--prediction_dir_name", type=str, help="directory name to save predictions")
        parser.add_argument("--test_validation_dir_name", type=str, help="directory name to save validation/test results")

    def before_instantiate_classes(self) -> None:
        """Makes some directories creation before any class instanciation."""
        subcommand = self.config.subcommand

        # generate directory name of the model checkpoint callback
        if self.config[subcommand]["model_checkpoint"]["dirpath"] == "":
            self.config[subcommand]["model_checkpoint"]["dirpath"] = os.path.join(
                self.config[subcommand]["trainer"]["logger"]["init_args"]["name"],
                self.config[subcommand]["checkpoint_dir_name"],
            )

        # generate directory name of the logger
        if self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"] == "":
            self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"] = os.path.join(
                self.config[subcommand]["trainer"]["logger"]["init_args"]["name"],
                self.config[subcommand]["log_dir_name"],
            )
        
        # generate directory name of the prediction directory
        if self.config[subcommand]["model"]["prediction_dir"] == "":
            self.config[subcommand]["model"]["prediction_dir"] = os.path.join(
                self.config[subcommand]["trainer"]["logger"]["init_args"]["name"],
                self.config[subcommand]["prediction_dir_name"],
            )
        
        # generate directory name of the test and validation directory
        if self.config[subcommand]["model"]["test_validation_dir"] == "":
            self.config[subcommand]["model"]["test_validation_dir"] = os.path.join(
                self.config[subcommand]["trainer"]["logger"]["init_args"]["name"],
                self.config[subcommand]["test_validation_dir_name"],
            )

        # creation of the directories
        os.makedirs(self.config[subcommand]["model_checkpoint"]["dirpath"], exist_ok=True)
        os.makedirs(self.config[subcommand]["trainer"]["logger"]["init_args"]["save_dir"], exist_ok=True)
        os.makedirs(self.config[subcommand]["model"]["prediction_dir"], exist_ok=True)
        os.makedirs(self.config[subcommand]["model"]["test_validation_dir"], exist_ok=True)


if __name__ == "__main__":
    cli = UNETRFinetuningCLI(model_class=SegmentationTask, datamodule_class=MSDDataModule,
                            trainer_class=pl.Trainer, description="UNETR finetuning CLI (pipeline test)")
