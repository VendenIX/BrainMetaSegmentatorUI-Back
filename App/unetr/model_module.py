"""Module that contains a Lightning module to easily perform
any operation with the Pytorch Lightning."""

from functools import partial
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from monai.data import decollate_batch, DataLoader
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.metrics.metric import CumulativeIterationMetric
from monai.transforms import AsDiscrete, AsDiscreted, Compose, Transform
from monai.utils.enums import MetricReduction
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.cli import instantiate_class
import torch

from meta.data.dataset import MetaSubset
from .dice_bce_loss import DiceBCELoss ###
from .networks.unetr import UNETR ###
from unetr.utilsUnetr.saver_logger_utils import ImageSaver, WandbLoggerUtils
from unetr.utilsUnetr.types import (
    ActionType,
    LabelColors,
    LabelNames,
    Metrics,
    PredictionSavingType,
    WandbResultLogging,
)
import numpy as np
import numpy

class SegmentationTask(pl.LightningModule):
    """Class that wraps the medical segmentation task as a PyTorch Lightning module.

    Attributes:
        model: Network/pretrained model used to be train and solve the task.
        backbone: Define a backbone from the used model for a possible finetuning.
        model_inferer: Sliding window for model inference.
        loss_fn: Loss function (if binary, `DiceBCELoss` is used, else the `DiceCELoss`)
        post_label: Post processer of the label.
        post_pred: Post processer of the model output.
        metrics: Dictionnary that contains all the used metrics to rate the model from different methods.
        max_epochs: Max number of iteration to fine-tune the model.
        image_saver: Image saver utils instance.
        val_test_logging_type: Type of logging for the validation and test stages.
        prediction_logging_type: Type of logging for the prediction stage.
        labels_names: Names of the labels.
        labels_colors: Colors of the labels.
        save_max_n_batches: Max number of batches to save.
        log_max_n_batches: Max number of batches to log.
    """

    def __init__(
        self,
        prediction_dir: str,
        test_validation_dir: str,
        pretrained_file_path: str,
        in_channels: int = 1,
        out_channels: int = 14,
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        new_out_channels: int = 1,
        number_of_blocks_to_tune: int = 1,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        infer_overlap: float = 0.5,
        max_epochs: int = 500,
        labels_names: LabelNames = None,
        labels_colors: LabelColors = None,
        smooth_dr: float = 1e-6,
        smooth_nr: float = 0.0,
        sw_batch_size: int = 1,
        use_bce_loss_when_binary_problem: bool = False,
        save_max_n_batches: Optional[int] = None,
        test_saving_type: List[PredictionSavingType] = [PredictionSavingType.NOTHING],
        prediction_saving_type: List[PredictionSavingType] = [PredictionSavingType.NOTHING],
        metrics: List[Metrics] = [Metrics.DICE, Metrics.HAUSDORFF_DISTANCE_95],
        log_max_n_batches: Optional[int] = None,
        val_test_logging_type: List[WandbResultLogging] = None,
        prediction_logging_type: List[WandbResultLogging] = None,
    ):
        """
        Arguments:
            prediction_dir: Directory to save prediction stage outputs.
            test_validation_dir: Directory to save validation/test model outputs.
            pretrained_file_path: Path of pretrained model.
            in_channels: Dimension of input channels.
            out_channels: Dimension of output channels.
            roi_size: Dimension of input image.
            new_out_channels: Dimension of the new output channels (for finetuning).
            number_of_blocks_to_tune: Number of blocks to tune (for finetuning).
            feature_size: Dimension of network feature size.
            hidden_size: Dimension of hidden layer.
            mlp_dim: Dimension of feedforward layer.
            num_heads: Number of attention heads.
            pos_embed: Position embedding layer type.
            norm_name: Feature normalization type and arguments.
            conv_block: Bool argument to determine if convolutional block is used.
            res_block: Bool argument to determine if residual block is used.
            dropout_rate: Fraction of the input units to drop.
            infer_overlap: Inference overlap of the sliding window.
            max_epochs: Max number of iteration to fine-tune the model.
            labels_names: Names of the labels.
            labels_colors: Colors of the labels.
            smooth_dr: A small constant added to the denominator to avoid nan.
            smooth_nr: A small constant added to the numerator to avoid zero.
            sw_batch_size: Size of the batch to process in the sliding window.
            use_bce_loss_when_binary_problem: Use the DiceBCELoss instead of DiceLoss when the problem is a binary segmentation.
            save_max_n_batches: Max number of batches to save.
            test_saving_type: Type of saving for the test stage.
            prediction_saving_type: Type of saving for the prediction stage.
            metrics: Type of metrics to use.
            log_max_n_batches: Max number of batches to log.
            val_test_logging_type: Type of logging for the validation and test stages.
            prediction_logging_type: Type of logging for the prediction stage.
        """
        super(SegmentationTask, self).__init__()
        
        self.model = UNETR.from_pretrained(
            torch.load(os.path.normpath(pretrained_file_path)), in_channels, 
            out_channels, roi_size, new_out_channels=new_out_channels,
            number_of_blocks_to_tune=number_of_blocks_to_tune,
            feature_size=feature_size, hidden_size=hidden_size,
            mlp_dim=mlp_dim, num_heads=num_heads, pos_embed=pos_embed, norm_name=norm_name,
            conv_block=conv_block, res_block=res_block, dropout_rate=dropout_rate,
        )
        self.backbone = torch.nn.Sequential(*self.model.backbone)
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=infer_overlap
        )

        if new_out_channels == 2:
            if use_bce_loss_when_binary_problem:
                self.loss_fn = DiceBCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=smooth_nr, smooth_dr=smooth_dr)
            else:
                self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=smooth_nr, smooth_dr=smooth_dr)
        else:
            self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=smooth_nr, smooth_dr=smooth_dr)
        self.post_label = Compose([AsDiscrete(to_onehot=True, num_classes=self.model.out_channels)])
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=True, num_classes=self.model.out_channels)])
        self._post_transforms = {}

        # metrics things
        self.metrics: Dict[str, CumulativeIterationMetric] = {}
        if Metrics.DICE in metrics:
            self.metrics[Metrics.DICE] = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=False)
            self.metrics[f"{Metrics.DICE}_without_bg"] = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=False)
        if Metrics.HAUSDORFF_DISTANCE_95 in metrics:
            self.metrics[Metrics.HAUSDORFF_DISTANCE_95] = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN, get_not_nans=False)
            self.metrics[f"{Metrics.HAUSDORFF_DISTANCE_95}_without_bg"] = HausdorffDistanceMetric(include_background=False, percentile=95, reduction=MetricReduction.MEAN, get_not_nans=False)
        assert len(self.metrics) != 0, "You need to have at least one metric to perform training"

        # some other utils classes or useful variables
        self.max_epochs = max_epochs
        self.save_max_n_batches = save_max_n_batches
        self.image_saver = ImageSaver(test_validation_dir, prediction_dir, test_saving_type, prediction_saving_type)
        
        self.log_max_n_batches = log_max_n_batches
        self.val_test_logging_type = val_test_logging_type or [WandbResultLogging.SEGMENTER]
        self.prediction_logging_type = prediction_logging_type or [WandbResultLogging.SEGMENTER]
        self.labels_names = labels_names or {0: "other", 1: "meta"}
        self.labels_colors = labels_colors or {0: (0, 0, 0), 1: (255, 0, 0)}
        self._logger_utils = None

        self.save_hyperparameters()
        
    @property
    def logger_utils(self) -> WandbLoggerUtils:
        """Property that returns a W&B logger utils instance.
        
        Returns:
            logger_utils: W&B logger utils instance.
        """
        if self._logger_utils is None:
            self._logger_utils = WandbLoggerUtils(self.trainer.logger, self.log, self.val_test_logging_type,
                                                  self.prediction_logging_type, self.labels_names, self.labels_colors)
        return self._logger_utils
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.forward(*args, **kwargs)
    
    def set_progress_bar_description(self, patient_id: Union[str, Sequence[str]], action_type: ActionType) -> None:
        """Sets a progress bar description to have a better vision
        of the training/validation/testing/prediction stage.
        
        Arguments:
            patient_id: ID(s) of the patient(s) that are currently passing in the model.
            action_type: Type of model action.
        """
        bar: TQDMProgressBar = self.trainer.progress_bar_callback
        desc = None
        if action_type == ActionType.TRAINING:
            desc = f"Epoch {self.current_epoch+1}/{self.max_epochs} -> {patient_id}"
        elif action_type == ActionType.TESTING:
            desc = f"Test inference -> {patient_id}"
        elif action_type == ActionType.VALIDATION:
            desc = f"Validation inference -> {patient_id}"
        elif action_type == ActionType.PREDICTION:
            desc = f"Prediction inference -> {patient_id}"

        bar.main_progress_bar.set_description_str(desc)
    
    def _get_post_transforms(self, dataloader: Optional[DataLoader] = None) -> Transform:
        """Gets the post transform associated to the dataloader.
        
        Arguments:
            dataloader: Dataloader to get the correct inverse transform.
        
        Returns:
            transform: Inverse transform after some processing.
        """
        if dataloader is None:
            return AsDiscreted(keys="pred", argmax=True)

        dataset: MetaSubset = dataloader.dataset # to get advanced IDE typing
        dataset_repr = str(dataset) # to have a representation of the dataset to easily retrieve it

        if dataset_repr in self._post_transforms:
            return self._post_transforms[dataset_repr]

        # final inverse transform
        self._post_transforms[dataset_repr] = dataset.get_inverse_transform()
        return self._post_transforms[dataset_repr]
    
    def post_process_data(self, dataloader: DataLoader, dict_: dict, input_: torch.Tensor, logits: torch.Tensor, 
                          target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process the data after inference.
        
        Arguments:
            dataloader: Dataloader to get the correct inverse transform.
            input: Image to predict.
            logits: Predicted logits.
            target: Ground truth mask.
        
        Returns:
            data: Post processed data (applying inverse transform).
        """
        inverse_transforms = self._get_post_transforms(dataloader)

        if type(dict_) == list:
            dict_=dict_[0]

        def seek_dict(dict_,key,inTransforms):
            for index in range (0,len(dict_[key])):
                dict_[key][index] = verif_cpu_values(dict_[key], index,inTransforms)
            return dict_[key]
        def verif_cpu_values(dict_, key, inTransforms):
            if "_transforms" in str(key) or inTransforms==True:
                inTransforms=True
                if type(dict_[key]) is list and "_transforms" not in key:
                    if type(dict_[key][0]) is str:
                        dict_[key]=dict_[key][0]
                    elif key!="center":
                        dict_[key]=tuple(seek_dict(dict_,key,inTransforms))
                if type(dict_[key]) is torch.Tensor and key!="box_end" and key!="box_start" and key!="old_affine":
                    dict_[key]=int(dict_[key].cpu()[0])
                    if dict_[key]==0:
                        dict_[key]=False
                    if dict_[key]==1:
                        dict_[key]=True
            if type(dict_[key]) is numpy.ndarray:
                dict_[key] = np.array([dict_[key]])
            if type(dict_[key]) is list:
                if type(dict_[key][0]) is str:
                    dict_[key]=dict_[key][0]
                else:
                    dict_[key] = seek_dict(dict_,key,inTransforms)
            if type(dict_[key]) is dict:
                dict_[key] = cpu_values(dict_[key],inTransforms)
            if type(dict_[key]) is torch.Tensor:
                dict_[key]=np.array(dict_[key].cpu()[0])
                if str(dict_[key])==str(False):
                    dict_[key]=False
                elif str(dict_[key])==str(True):
                    dict_[key]=True
                elif key=="patch_index":
                    dict_[key]=int(dict_[key])
            return dict_[key]
        def cpu_values(dict_,inTransforms=False):
            for key, value in dict_.items():
                dict_[key] = verif_cpu_values(dict_, key,inTransforms)
            return dict_

        def applyTransformation(input_, target, logits, inverse_transforms):
            post_logits=[]
            dict_["image"] = input_.cpu()
            dict_["pred"] = logits.cpu
            dict_["label"] = target.cpu() if target is not None else torch.zeros_like(logits)
            post_dict = inverse_transforms(dict_)

            dict_["label"] = logits[0][None]
            post_logits.append(np.array(inverse_transforms(dict_)["label"]))
            dict_["label"] = logits[1][None]
            post_logits.append(np.array(inverse_transforms(dict_)["label"]))
            post_logits=np.array(post_logits)
            post_logits=post_logits.reshape(post_logits.shape[0], post_logits.shape[2], post_logits.shape[3], post_logits.shape[4])
            return post_dict["image"], post_dict["label"], post_logits
        
        dict_ = cpu_values(dict_)
        if len(input_.shape)==5 and input_.shape[0]==1:
            size = input_.shape
            input_ = input_.reshape((size[1], size[2], size[3], size[4]))
            #logits = torch.argmax(logits, dim=1, keepdim=True)
            logits = logits.reshape((2, size[2], size[3], size[4]))
            target = target.reshape((size[1], size[2], size[3], size[4]))
            val_input, val_target, val_logits = applyTransformation(input_, target, logits, inverse_transforms)
            val_input, val_target, val_logits = val_input[None], val_target[None], val_logits[None]
        elif input_.shape[0]!=1:
            val_input=[]
            val_target=[]
            val_logits=[]
            #logits = torch.argmax(logits, dim=1, keepdim=True)
            #input_.shape[0]):
            for i in range (0, 2):
                val_input0, val_target0, val_logits0 = applyTransformation(input_[i], target[i], logits[i], inverse_transforms)
                val_input.append(np.array(val_input0))
                val_target.append(np.array(val_target0))
                val_logits.append(np.array(val_logits0))
            val_input, val_target, val_logits = val_input, val_target, val_logits
        else:
            #logits = torch.argmax(logits.cpu(), dim=0, keepdim=True)
            val_input, val_target, val_logits = applyTransformation(input_, target, logits, inverse_transforms)
        return torch.tensor(val_input, requires_grad=True), torch.tensor(val_target, requires_grad=True), torch.tensor(val_logits, requires_grad=True)
    
    def _get_values_from_batch(self, batch):
        """Extracts values from the batch.
        
        Arguments:
            batch: Batch of data to extract.
        
        Returns:
            image: Image to predict.
            label: Associated label to image.
            patient_id: Id of the associated patient.
            has_meta: Patient has meta or not.
        """
        if isinstance(batch, dict):
            return batch["image"], batch["label"], batch["patient_id"], batch["has_meta"], batch["dict_"]
        
        return batch
    
    def training_step(self, batch, batch_idx):
        """Operates on a single batch of data from the train set.
        In this step, predictions are generated and metrics are computed to get the average train accuracies.

        Arguments:
            batch: The output of your `~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
        """
        data, target, patient_id, _ , dict_ = self._get_values_from_batch(batch)
        self.set_progress_bar_description(list(set(patient_id)), ActionType.TRAINING) # set() because of multiple patches by samples

        # realize predictions and compute loss
        logits = self.model(data)
        data, target, logits = self.post_process_data(self.trainer.val_dataloaders[0], dict_, data, logits, target)
        loss = self.loss_fn(logits, target)

        # log the training loss
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        
        return loss
    
    def on_train_epoch_end(self, *arg, **kwargs) -> None:
        """Called in the train loop at the very end of the epoch.
        Only the learning rate is logged to get an eye on this during training.

        To get the learning rate, we need to interact with learning rate
        schedulers because we can't access current learning rate through
        the optimizers instances.

        Arguments:
            *args: Ignored.
            **kwargs: Ignored.
        """
        schedulers = self.lr_schedulers()

        # we don't handle multiple schedulers or if there is no scheduler
        if schedulers is None or isinstance(schedulers, list):
            return
        
        self.log("learning_rate", schedulers.get_lr()[0])
    
    def on_validation_epoch_start(self) -> None:
        """Called in the validation loop at the very beginning of the epoch.
        Only the validation table is initialized.
        """
        self.logger_utils.init_tables(ActionType.VALIDATION)
    
    def validation_step(self, batch, batch_idx):
        """Operates on a single batch of data from the validation set.
        In this step, predictions are generated and metrics are computed to get the average validation accuracies.

        Arguments:
            batch: The output of your `~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
        """
        data, target, patient_id, has_meta, dict_ = self._get_values_from_batch(batch)

        # realize predictions, compute loss and make post prediction transforms
        logits = self.model_inferer(data)
        loss = self.loss_fn(logits, target)

        val_outputs = [self.post_pred(i) for i in decollate_batch(logits)]
        val_labels = [self.post_label(i) for i in decollate_batch(target)]

        # compute metrics for the predicted samples
        for key in self.metrics.keys():
            self.metrics[key](y_pred=val_outputs, y=val_labels)
        
        # log the validation loss
        self.log("val_loss", loss, on_epoch=True, on_step=False)

        # prepare data and add them to the prediction table that will be logged at the epoch end
        # data, target, preds = self.post_process_data(self.trainer.val_dataloaders[0], data, logits, target)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        for ii, single_patient_id in enumerate(patient_id):

            if (self.log_max_n_batches and self.log_max_n_batches < batch_idx) or self.log_max_n_batches is None:
                self.logger_utils.log_or_add_data(self.current_epoch, self.trainer.sanity_checking, 
                    ActionType.VALIDATION, single_patient_id, has_meta[ii], data[ii], preds[ii], target[ii])
        
        return loss
    
    def on_validation_epoch_end(self, *arg, **kwargs) -> None:
        """Called in the validation loop at the very end of the epoch.
        The validation metrics and the validation table are logged and reset after logging.

        Arguments:
            *args: Ignored.
            **kwargs: Ignored.
        """
        # log metrics and validation table
        if Metrics.DICE in self.metrics:
            self.log("dice_val_acc (higher is best)", self.metrics[Metrics.DICE].aggregate())
            self.log("dice_val_acc w/out bg (higher is best)", self.metrics[f"{Metrics.DICE}_without_bg"].aggregate())
        if Metrics.HAUSDORFF_DISTANCE_95 in self.metrics:
            self.log("hd95_val (less is best)", self.metrics[Metrics.HAUSDORFF_DISTANCE_95].aggregate())
            self.log("hd95_val w/out bg (less is best)", self.metrics[f"{Metrics.HAUSDORFF_DISTANCE_95}_without_bg"].aggregate())
        self.logger_utils.log_table(ActionType.VALIDATION)

        # reset metrics and validation table
        for key in self.metrics.keys():
            self.metrics[key].reset()
        self.logger_utils.init_tables(ActionType.VALIDATION)
    
    def on_predict_epoch_start(self) -> None:
        """Called in the predict loop at the very beginning of the epoch.
        Only the predict table is initialized.
        """
        self.logger_utils.init_tables(ActionType.PREDICTION)
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Operates on a single batch of data from the predict set.
        In this step, predictions are generated and are logged and saved corresponding to init config.

        Arguments:
            batch: The output of your `~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_id: The index of the dataloader that produced this batch.
                (only if multiple predict dataloaders used).
        """
        data, _, patient_id, has_meta, dict_  = self._get_values_from_batch(batch)
        self.set_progress_bar_description(patient_id, ActionType.PREDICTION)

        # realize predictions and make post prediction transforms
        logits = self.model_inferer(data)

        # prepare data and add them to the predict table that will be logged at the epoch end
        # and save at step
        # data, _, preds = self.post_process_data(self.trainer.val_dataloaders[0], data, logits)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        for ii, single_patient_id in enumerate(patient_id):

            if (self.save_max_n_batches and self.save_max_n_batches < batch_idx) or self.save_max_n_batches is None:
                self.image_saver.save(data[ii,...], None, preds[ii,...], ii, ActionType.PREDICTION)
            if (self.log_max_n_batches and self.log_max_n_batches < batch_idx) or self.log_max_n_batches is None:
                self.logger_utils.log_or_add_data(self.current_epoch, self.trainer.sanity_checking, 
                    ActionType.PREDICTION, single_patient_id, has_meta[ii], data[ii], preds[ii])
    
    def on_predict_epoch_end(self, *arg, **kwargs) -> None:
        """Called in the test loop at the very end of the epoch.
        The test metrics and the test table are logged and reset after logging.

        Arguments:
            *args: Ignored.
            **kwargs: Ignored.
        """
        # log predict table
        self.logger_utils.log_table(ActionType.PREDICTION)
        
        # reset predict table
        self.logger_utils.init_tables(ActionType.PREDICTION)
    
    def on_test_epoch_start(self) -> None:
        """Called in the test loop at the very beginning of the epoch.
        Only the test table is initialized.
        """
        self.logger_utils.init_tables(ActionType.TESTING)
    
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Operates on a single batch of data from the test set.
        In this step, predictions are generated and metrics are computed to get the average test accuracies.

        Arguments:
            batch: The output of your `~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_id: The index of the dataloader that produced this batch.
                (only if multiple test dataloaders used).
        """
        data, target, patient_id, has_meta, dict_  = self._get_values_from_batch(batch)
        self.set_progress_bar_description(patient_id, ActionType.TESTING)
        
        # realize predictions, compute loss and make post prediction transforms
        logits = self.model_inferer(data)
        loss = self.loss_fn(logits, target)

        val_outputs = [self.post_pred(i) for i in decollate_batch(logits)]
        val_labels = [self.post_label(i) for i in decollate_batch(target)]

        # compute metrics for the predicted samples
        for key in self.metrics.keys():
            self.metrics[key](y_pred=val_outputs, y=val_labels)
        
        # log the test loss
        self.log("test_loss", loss, on_epoch=True, logger=True, on_step=False)
        
        # prepare data and add them to the test table that will be logged at the epoch end
        # and save at step
        # data, target, preds = self.post_process_data(self.trainer.val_dataloaders[0], data, logits, target)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        for ii, single_patient_id in enumerate(patient_id):

            if (self.save_max_n_batches and self.save_max_n_batches < batch_idx) or self.save_max_n_batches is None:
                self.image_saver.save(data[ii,...], target[ii,...], preds[ii,...], ii, ActionType.TESTING)
            if (self.log_max_n_batches and self.log_max_n_batches < batch_idx) or self.log_max_n_batches is None:
                self.logger_utils.log_or_add_data(self.current_epoch, self.trainer.sanity_checking, 
                    ActionType.TESTING, single_patient_id, has_meta[ii], data[ii], preds[ii], target[ii])
    
    def on_test_epoch_end(self, *arg, **kwargs) -> None:
        """Called in the test loop at the very end of the epoch.
        The test metrics and the test table are logged and reset after logging.

        Arguments:
            *args: Ignored.
            **kwargs: Ignored.
        """
        # log metrics and test table
        if Metrics.DICE in self.metrics:
            self.log("dice_test_acc (higher is best)", self.metrics[Metrics.DICE].aggregate())
            self.log("dice_test_acc w/out bg (higher is best)", self.metrics[f"{Metrics.DICE}_without_bg"].aggregate())
        if Metrics.HAUSDORFF_DISTANCE_95 in self.metrics:
            self.log("hd95_test (less is best)", self.metrics[Metrics.HAUSDORFF_DISTANCE_95].aggregate())
            self.log("hd95_test w/out bg (less is best)", self.metrics[f"{Metrics.HAUSDORFF_DISTANCE_95}_without_bg"].aggregate())
        self.logger_utils.log_table(ActionType.TESTING)

        # reset metrics and test table
        for key in self.metrics.keys():
            self.metrics[key].reset()
        self.logger_utils.init_tables(ActionType.TESTING)
    
    def configure_optimizers(self) -> Any:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            optimizers: List of optimizers.
            schedulers: List of learning rate schedulers.
        """
        # instantiate class chosen in the CLI
        optimizer = instantiate_class(self.parameters(), self.hparams.optimizer)
        scheduler = instantiate_class(optimizer, self.hparams.lr_scheduler)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        """Overrides the way of calling the learning rate scheduler step.

        Arguments:
            scheduler: Learning rate scheduler.
            optimizer_idx: Index of the optimizer associated with this scheduler. Ignored.
            metric: Value of the monitor used for schedulers like `ReduceLROnPlateau`. Ignored.
        """
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step(epoch=self.current_epoch)


def convert_model_to_torchscript(module: pl.LightningModule, output_dir: str, model_filename: str = "model.pt"):
    """Converts a model to a torchscript for a future use.
    
    Arguments:
        module: Model module to convert.
        output_dir: Directory to save the converted model.
        model_filename: Filename of the output model.
    """
    module.to_torchscript(os.path.join(output_dir, model_filename))
