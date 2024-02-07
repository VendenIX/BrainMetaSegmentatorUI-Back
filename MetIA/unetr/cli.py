"""Module that contains the fine-tuning CLI.

Created with the Pytorch LightningCLI, it allows
to easily choose any hyperparameters and
to quickly manage the stage type.
"""

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning, EarlyStopping, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser

from model_module import SegmentationTask
from unetr.utilsUnetr.data_module import MetaDataModule


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
    cli = UNETRFinetuningCLI(model_class=SegmentationTask, 
                            datamodule_class=MetaDataModule,
                            trainer_class=pl.Trainer, 
                            description="UNETR finetuning CLI")
    
