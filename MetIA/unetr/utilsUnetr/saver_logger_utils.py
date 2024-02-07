"""This module defines some utilitary classes to make some
code more easily usable from other classes."""

from functools import partial
import os
from typing import Callable, List, Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
from monai.networks.utils import one_hot
import nibabel as nib
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from unetr.utilsUnetr.tensor_utils import TensorUtils
from unetr.utilsUnetr.transforms import SampleNormalizer
from unetr.utilsUnetr.types import ActionType, LabelColors, LabelNames, PredictionSavingType, WandbResultLogging


class ImageSaver:
    """Class utils to easily handle image saving.

    Attributes:
        validation_test_dir: Directory to save validation/test model outputs.
        prediction_dir: Directory to save prediction stage outputs.
        validation_test_saving_type: Type of saving for the test stage.
        prediction_saving_type: Type of saving for the prediction stage.
    """
    def __init__(self, validation_test_dir: str, prediction_dir: str,
                 validation_test_saving_type: List[PredictionSavingType],
                 prediction_saving_type: List[PredictionSavingType]) -> None:
        """
        Arguments:
            validation_test_dir: Directory to save validation/test model outputs.
            prediction_dir: Directory to save prediction stage outputs.
            validation_test_saving_type: Type of saving for the test stage.
            prediction_saving_type: Type of saving for the prediction stage.
        """
        PredictionSavingType.verify(prediction_saving_type)
        PredictionSavingType.verify(validation_test_saving_type)

        self.validation_test_dir = validation_test_dir
        self.validation_test_saving_type = validation_test_saving_type
        self.prediction_dir = prediction_dir
        self.prediction_saving_type = prediction_saving_type
    
    def get_dir(self, action_type: ActionType) -> str:
        """Gets the associted saving directory to the `action_type`.
        
        Arguments:
            action_type: Type of model action.
        
        Returns:
            directory: Saving directory associated to the action.
        """
        if action_type == ActionType.PREDICTION:
            return self.prediction_dir
        
        if action_type in (ActionType.TESTING, ActionType.VALIDATION):
            return self.validation_test_dir
        
        raise ValueError("specified action_type is invalid")
    
    def get_saving_type(self, action_type: ActionType) -> str:
        """Gets the associted saving type to the `action_type`.
        
        Arguments:
            action_type: Type of model action.
        
        Returns:
            saving_type: Type of saving associated to the action.
        
        Raises:
            ValueError: When the specified `action_type` is invalid.
        """
        if action_type == ActionType.PREDICTION:
            return self.prediction_saving_type
        
        if action_type in (ActionType.TESTING, ActionType.VALIDATION):
            return self.validation_test_saving_type
        
        raise ValueError("specified action_type is invalid")
    
    def save(self, input: torch.Tensor, label: Union[torch.Tensor, None],
             preds: torch.Tensor, patient_id: str, action_type: ActionType) -> None:
        """Saves an input image and label/preds according to the `action_type`.
        
        Attributes:
            input: Image to predict.
            label: Ground truth mask.
            preds: Predicted mask.
            patient_id: ID of the patient to easily retrieve saved slices.
            action_type: Type of model action.
        """
        directory = self.get_dir(action_type)
        saving_type = self.get_saving_type(action_type)

        if PredictionSavingType.NOTHING:
            pass # save no slices
        elif PredictionSavingType.RANDOM in saving_type: # save a random slice as 2D image
            idx = np.random.randint(0, input.shape[-1])
            self.save_2d_slice(input, label, preds, idx, patient_id, directory)
        elif PredictionSavingType.ALL in saving_type: # save all slices as 2D images
            for slice_idx in range(input.shape[-1]):
                self.save_2d_slice(input, label, preds, slice_idx, patient_id, directory)
        else: # save specific slices as 2D images
            indices = PredictionSavingType.slices_to_save(saving_type)
            for slice_idx in indices:
                if indices < input.shape[-1]:
                    warnings.warn(f"the specified slice {slice_idx} doesn't exists for patient {patient_id}")
                self.save_2d_slice(input, label, preds, slice_idx, patient_id, directory)
        
        # save the entire 3D image in Nifti format
        if PredictionSavingType.AS_NIFTI in saving_type:
            self.save_3d_image(preds, patient_id, directory, action_type)
    
    @staticmethod
    def get_action_type_as_str(action_type: ActionType) -> str:
        """Gets the associted key to the `action_type`.
        
        Arguments:
            action_type: Type of model action.
        
        Returns:
            key: Associated key.
        
        Raises:
            ValueError: When the specified `action_type` is invalid.
        """
        if action_type.VALIDATION:
            return "val"
        
        if action_type.TESTING:
            return "test"
        
        if action_type.PREDICTION:
            return "predict"
        
        if action_type.TRAINING:
            return "train"
        
        raise ValueError("specified type is invalid")

    @classmethod
    def save_2d_slice(cls, input: torch.Tensor, label: Union[torch.Tensor, None],
                      preds: torch.Tensor, slice_idx: int, patient_id: str,
                      saving_dir: str) -> None:
        """Saves a slice.
        
        According to the passed arguments, the slice is saved
        with the input data, prediction and label if is not `None`.

        Arguments:
            input: Image to predict.
            label: Ground truth mask.
            preds: Predicted mask.
            slice_idx: Index of the slice to save.
            patient_id: ID of the patient to easily retrieve saved slices.
            saving_dir: Directory to save the images.
        """
        number_of_images = 2 if label is None else 3

        plt.figure("check", (6 * number_of_images, 6))

        plt.subplot(1, number_of_images, 1)
        plt.title("image")
        plt.imshow(input[0, :, :, slice_idx], cmap="gray")

        if label is not None:
            plt.subplot(1, number_of_images, 2)
            plt.title("label")
            plt.imshow(label[0, :, :, slice_idx])
        
        plt.subplot(1, number_of_images, number_of_images)
        plt.title("output")
        plt.imshow(preds[0, :, :, slice_idx])

        plt.savefig(os.path.join(saving_dir, f"{patient_id}.{slice_idx}.jpg"))
    
    @classmethod
    def save_3d_image(cls, array: torch.Tensor, patient_id: str, saving_dir: str,
                      action_type: ActionType) -> None:
        """Saves a 3D image in the Nifti format.

        It saves only the first mask. Please you want to save multiple masks,
        make a for loop that iterating the call with different masks.

        Arguments:
            array: Image to save.
            filepath: Saving file path.
        
        See also:
            nib.Nifti1Image, nib.save
        """
        assert 3 <= len(array.shape) <= 5, "array should be have one the following formats: DHW, MDHW or BMDHW where M is the wanted mask."

        # get only the first mask
        if len(array.shape) == 5:
            array = array[0]
        if len(array.shape) == 4 and array.shape[0] != 1:
            array = one_hot(array, array.max(), dim=0)
        
        for ii in range(array.shape[0]):
            image = nib.Nifti1Image(array, np.eye(4))
            nib.save(image, os.path.join(saving_dir, f"{patient_id}.{cls.get_action_type_as_str(action_type)}.{ii}.nii.gz"))


class WandbLoggerUtils:
    """Utils class that help to log tables to W&B panel
    view from a PyTorch Lightning Module.
    
    Attributes:
        normalizer: Transform used at different locations to normalize data.
        logger_ok: Is the logger is a correct W&B logger instance.
        logger: W&B logger instance.
        log_func: Logging function of the main Lightning module.
        validation_test_logging_type: Type of logging for the validation and test stages.
        prediction_logging_type: Type of logging for the prediction stage.
        labels_names: Names of the labels.
        labels_colors: Colors of the labels.
        validation_table: W&B validation table that will be logged.
        test_table: W&B test table that will be logged.
        prediction_table: W&B prediction table that will be logged.
    """
    normalizer = SampleNormalizer()

    def __init__(self, logger: WandbLogger, log_func: Callable,
                 validation_test_logging_type: List[WandbResultLogging],
                 prediction_logging_type: List[WandbResultLogging],
                 labels_names: LabelNames,
                 labels_colors: LabelColors) -> None:
        """
        Arguments:
            logger: W&B logger instance.
            log_func: Logging function of the main Lightning module.
            validation_test_logging_type: Type of logging for the validation and test stages.
            prediction_logging_type: Type of logging for the prediction stage.
            labels_names: Names of the labels.
            labels_colors: Colors of the labels.
        """
        # runtime type checking
        self.logger_ok = True
        if not isinstance(logger, WandbLogger):
            self.logger_ok = False

        WandbResultLogging.verify(prediction_logging_type)
        WandbResultLogging.verify(validation_test_logging_type)

        self.logger = logger
        self.log_func = log_func
        self.validation_test_logging_type = validation_test_logging_type
        self.prediction_logging_type = prediction_logging_type
        self.labels_names = labels_names
        self.labels_colors = labels_colors

        self.validation_table = None
        self.test_table = None
        self.prediction_table = None
    
    def _transform_image(self, array: torch.Tensor) -> torch.Tensor:
        """Transforms the array to a normalized image between
        0 and 255 (to the standard pixel value range).
        
        Arguments:
            array: Data to process.
        
        Returns:
            array: Normalized array between 0 and 255.
        """
        return self.normalizer(array, dtype=torch.float) * 255
    
    def _get_attributes_from_action_type(self, action_type: ActionType) -> Tuple[str, wandb.Table, List[WandbResultLogging]]:
        """Retrieves associated attributes to `action_type` argument.
        
        Arguments:
            action_type: Type of model action.
        
        Returns:
            key: Key associated to the action.
            table: W&B table instance.
            logging_type: Type of table to log.
        
        Raises:
            ValueError: When the specified `action_type` is invalid.
        
        See also:
            ImageSaver.get_action_type_as_str
        """
        key = ImageSaver.get_action_type_as_str(action_type)
        if action_type == ActionType.VALIDATION:
            return key, self.validation_table, self.validation_test_logging_type
        
        if action_type == ActionType.TESTING:
            return key, self.test_table, self.validation_test_logging_type
        
        if action_type == ActionType.PREDICTION:
            return key, self.prediction_table, self.prediction_logging_type
            
        raise ValueError("specified type is invalid")

    def _generate_patient_id_string(self, patient_id: str, has_meta: bool) -> str:
        """Generates the patient ID string for W&B tables.
        
        Arguments:
            patient_id: ID of the patient in the database.
            has_meta: Boolean that says if the patient CT scan contains a meta or not.
        
        Returns:
            string: Represents a patient with a meta or not.
        """
        return f"{patient_id} ({'yes' if has_meta else 'no'})"
    
    def init_tables(self, action_type: Optional[ActionType] = None) -> None:
        """Initializes the W&B tables according to the `action_type` argument.

        If `action_type=None`, all tables are initialized.
        
        Arguments:
            action_type: Type of model action for that we want a table initialization.
        
        See also:
            wandb.Table
        """
        if (action_type is None or action_type == ActionType.VALIDATION) and \
            WandbResultLogging.NOTHING not in self.validation_test_logging_type and \
            WandbResultLogging.LOG_AS_TABLE in self.validation_test_logging_type:
                self.validation_table = wandb.Table(columns=WandbResultLogging.init_columns_names(self.validation_test_logging_type, without_target=False))
        if (action_type is None or action_type == ActionType.TESTING) and \
            WandbResultLogging.NOTHING not in self.validation_test_logging_type and \
            WandbResultLogging.LOG_AS_TABLE in self.validation_test_logging_type:
                self.test_table = wandb.Table(columns=WandbResultLogging.init_columns_names(self.validation_test_logging_type, without_target=False))
        if (action_type is None or action_type == ActionType.PREDICTION) and \
            WandbResultLogging.NOTHING not in self.prediction_logging_type and \
            WandbResultLogging.LOG_AS_TABLE in self.prediction_logging_type:
                self.prediction_table = wandb.Table(columns=WandbResultLogging.init_columns_names(self.prediction_logging_type, without_target=True))
        
        torch.cuda.empty_cache()

    def _masked_image(self, slice_idx: int, base_image: np.ndarray, 
                     label_mask: Optional[np.ndarray] = None, 
                     pred_mask: Optional[np.ndarray] = None) -> wandb.Image:
        """Returns a W&B image to populate slider with masked images.
        
        Arguments:
            slice_idx: Index of the slice to display masks.
            base_image: Image to put masks on it.
            label_mask: Ground truth masks.
            pred_mask: Predicted masks.
        
        Returns:
            image: W&B image that contains the `base_image` with label and/or prediction masks.
        """
        masks = {}
        if label_mask is not None: # add ground truth mask if exists
            masks["ground truth"] = {
                "mask_data" : label_mask,
                "class_labels" : self.labels_names,
            }
        if pred_mask is not None: # add predicted mask if exists
            masks["prediction"] = {
                "mask_data" : pred_mask,
                "class_labels" : self.labels_names,
            }

        return wandb.Image(base_image, masks=masks, caption=f"Slice: {slice_idx}")

    def _get_3d_image_as_slider(self, input_: torch.Tensor, pred: torch.Tensor = None, 
            label: torch.Tensor = None, return_imgs: bool = True
        ) -> Tuple[List[wandb.Image], List[wandb.Image], List[wandb.Image]]:
        """Returns arrays as slider and segmentation masks inside the W&B view panel.

        It assumes that the tensors is in the CPU.
        
        Arguments:
            input_: Input image to predict.
            pred: Predicted masks.
            label: Ground truth masks.
            return_imgs: If enable, the raw images are returned.
        
        Returns:
            wandb_img_logs: 3D image. Only if ``return_imgs=True``
            wandb_pred_logs: Masked 3D image with the prediction mask.
            wandb_label_logs: Masked 3D image with the ground truth mask.
        
        See also:
            _masked_image
        """
        wandb_img_logs, wandb_pred_logs, wandb_label_logs = [], [], []

        # some assertion to verify shapes
        assert input_.shape == pred.shape, "Shapes need to be the same"
        if label is not None:
            assert input_.shape == label.shape, "Shapes need to be the same"
        
        input_, pred, label = TensorUtils.convert_to_numpy_array(input_, pred, label)
        input_, pred, label = TensorUtils.convert_to(input_, pred, label, dtype=np.uint8)
        
        # some reshaping to have a format to simplify the computations
        if len(input_.shape) > 3:
            if len(input_.shape) == 5:
                assert input_.shape[0] == 1 and input_.shape[1] == 1, "batch and channels dim need to be equal to 1"
                input_ = input_.reshape(input_.shape[2:])
                pred = pred.reshape(pred.shape[2:])
                if label is not None:
                    label = label.reshape(label.shape[2:])
            elif len(input_.shape) == 4:
                assert input_.shape[0] == 1, "batch dim need to be equal to 1"
                input_ = input_.reshape(input_.shape[1:])
                pred = pred.reshape(pred.shape[1:])
                if label is not None:
                    label = label.reshape(label.shape[1:])
        
        # create a masked image for each 3D image slices
        for slice_idx in range(input_.shape[-1]):
            img = input_[:,:,slice_idx]

            wandb_img_logs.append(wandb.Image(img, caption=f"Slice: {slice_idx}"))

            # make predicted and ground truth masked images
            wandb_pred_logs.append(self._masked_image(slice_idx, img, None, pred[...,slice_idx]))
            if label is not None:
                wandb_label_logs.append(self._masked_image(slice_idx, img, label[...,slice_idx]))
        
        if return_imgs:
            return wandb_img_logs, wandb_pred_logs, wandb_label_logs
        
        return wandb_pred_logs, wandb_label_logs

    def _get_3d_image_as_video(self, array: torch.Tensor, caption: str = None, fps: int = 4,
                               normalized: bool = False, is_mask: bool = False) -> wandb.Video:
        """Returns an array as a W&B video in the view panel.

        It assumes that the tensor is in the CPU.
        
        Arguments:
            array: Array to convert to a video.
            caption: Name of the video.
            fps: Number of frames per second to view.
            normalized: The fact that `array` has already been normalized.
            is_mask: Is the `array` a mask.
        
        Returns:
            video: `array` in the W&B video format.
        """
        new_array = array

        # normalize array if has not been normalized and if is not a mask
        if not normalized and not is_mask:
            new_array = self._transform_image(array)

        # generate colors for each label
        if is_mask:
            # to avoid write same code for another dimension
            dim = (2 if len(array.shape) == 5 else 1)
            assert array.shape[dim] == 1, f"Need to have only one channel. If you have more than one, you can use `torch.argmax(array, dim={dim}, keepdim=False)` before calling this function."

            # create a new array with 3 channels (R,G,B)
            new_array = torch.zeros(*array.shape[:dim], 3, *array.shape[-2:])
            for label in torch.unique(array):
                if dim == 1:
                    pix_label = array[:,0,...] == label
                    new_array[pix_label] += torch.tensor(self.labels_colors[label]) * array[pix_label]
                else:
                    pix_label = array[:,:,0,...] == label
                    new_array[pix_label] += torch.tensor(self.labels_colors[label]) * array[pix_label]
        
        new_array = TensorUtils.convert_to_numpy_array(new_array)
        new_array = TensorUtils.convert_to(new_array, dtype=np.uint8)
        return wandb.Video(new_array, caption=caption, fps=fps)
    
    def log_or_add_data(self, current_epoch: int, sanity_checking: bool, action_type: ActionType,
                        patient_id: str, has_meta: bool, input_: torch.Tensor, pred: torch.Tensor,
                        label: Optional[torch.Tensor] = None, fps: int = 4) -> None:
        """Add data to the correct table according to `action_type` if is a table.
        If it's not a table, directly log the data in a specific key in W&B panel.
        
        Arguments:
            current_epoch: Current epoch number.
            sanity_checking: Is the trainer in the sanity check.
            action_type: Type of model action for that we want log.
            patient_id: ID of the patient in the database.
            has_meta: Boolean that says if the patient CT scan contains a meta or not.
            input_: Input image to predict.
            pred: Predicted masks.
            label: Ground truth masks.
            fps: Number of frames per second to view.
        """
        if not self.check_can_log(action_type):
            return
        
        key, _, logging_type = self._get_attributes_from_action_type(action_type)

        is_table = WandbResultLogging.LOG_AS_TABLE in logging_type
        if is_table:
            self._add_data_to_table(
                current_epoch=current_epoch,
                sanity_checking=sanity_checking,
                action_type=action_type,
                patient_id=patient_id,
                has_meta=has_meta,
                input_=input_,
                pred=pred,
                label=label,
                fps=fps
            )
            return

        data_key = f"{key}_{patient_id}"
        d = {}
        if WandbResultLogging.VIDEO in logging_type:
            input_ = self._transform_image(input_)
            videos = self._get_data_as_videos(input_, pred, label, normalized=True, fps=fps)
            d.update({
                "prediction": videos[1],
            })
            if label is not None:
                d.update({"label": videos[2]})
        elif WandbResultLogging.SEGMENTER in logging_type:
            slicers = self._get_3d_image_as_slider(input_, pred, label, return_imgs=True)
            d.update({
                "prediction": slicers[1],
            })
            if label is not None:
                d.update({"label": slicers[2]})
        self.log_func(data_key, d)

    def _add_data_to_table(self, current_epoch: int, sanity_checking: bool, action_type: ActionType,
                           patient_id: str, has_meta: bool, input_: torch.Tensor, pred: torch.Tensor,
                           label: Optional[torch.Tensor] = None, fps: int = 4) -> None:
        """Add data to the correct table according to `action_type`.
        
        Arguments:
            current_epoch: Current epoch number.
            sanity_checking: Is the trainer in the sanity check.
            action_type: Type of model action for that we want log.
            patient_id: ID of the patient in the database.
            has_meta: Boolean that says if the patient CT scan contains a meta or not.
            input_: Input image to predict.
            pred: Predicted masks.
            label: Ground truth masks.
            fps: Number of frames per second to view.
        """
        # normalize the base image
        input_ = self._transform_image(input_)

        # get the correct logging type and table according to the `action_type`
        _, table, logging_type = self._get_attributes_from_action_type(action_type)
    
        # get correct epoch number
        epoch = current_epoch if not sanity_checking else "Sanity check"
        
        if WandbResultLogging.VIDEO in logging_type:
            func = partial(self._get_data_as_videos, normalized=True, fps=fps)

            # preprocess the data for video
            pred = self._transform_image(pred)
            if label is not None:
                label = self._transform_image(label)
        else:
            func = partial(self._get_3d_image_as_slider, return_imgs=False)

            # preprocess the data for slider
            pred = pred.to(torch.float32)
            if label is not None:
                label = label.to(torch.float32)
        
        # get the data in a good shape
        if len(input_.shape) == 5:
            input_, pred = input_[0], pred[0]
            if label is not None:
                label = label[0]
        
        # move to cpu to avoid CUDA out of memory error
        input_, pred, label = TensorUtils.move_tensors_to_cpu(input_, pred, label)
        TensorUtils.clear_gpu_memory()
        
        # add data to table
        table.add_data(epoch, self._generate_patient_id_string(patient_id, has_meta), *func(input_, pred, label))

    def _get_data_as_videos(self, input_: torch.Tensor, pred: torch.Tensor = None, 
                            label: torch.Tensor = None, normalized: bool = True, 
                            fps: int = 4) -> List[wandb.Video]:
        """Returns the data as a list of W&B videos.
        
        Attributes:
            input_: Input image to predict.
            pred: Predicted masks.
            label: Ground truth masks.
            normalized: Is the input and label images normalized before.
            fps: Number of frames per second to view.
        
        Returns:
            videos: Input, prediction and ground truth 3D images as videos.
        """
        returns = []

        returns.append(self._get_3d_image_as_video(input_, fps=fps, normalized=normalized))
        returns.append(self._get_3d_image_as_video(pred, fps=fps, is_mask=True))

        if label is not None:
            returns.append(self._get_3d_image_as_video(label, fps=fps, normalized=normalized))
        
        return returns

    def log_table(self, action_type: ActionType) -> None:
        """Logs the table to the W&B interface through their API.
        
        Arguments:
            action_type: Type of model action for that we want log.
        
        See also:
            WandbLogger.log_table, check_can_log
        """
        if not self.check_can_log(action_type):
            return
        
        if action_type in (ActionType.VALIDATION, ActionType.TESTING) and \
            WandbResultLogging.LOG_AS_TABLE not in self.validation_test_logging_type:
            return
        if action_type == ActionType.PREDICTION and \
            WandbResultLogging.LOG_AS_TABLE not in self.prediction_logging_type:
            return
            
        sub_key, table, _ = self._get_attributes_from_action_type(action_type)
        self.logger.log_table(key=f"{sub_key}_table", columns=table.columns, data=table.data)

    def check_can_log(self, action_type: ActionType) -> bool:
        """Checks if we can log with W&B.
        
        Arguments:
            action_type: Type of model action for that we want log.
        
        Returns:
            value: If we can log to W&B.
        """
        # if is not a W&B logger
        if not self.logger_ok:
            return False

        # if we are training
        if action_type == ActionType.TRAINING:
            return False
        
        # if the table logging is disabled
        if action_type == ActionType.PREDICTION:
            if WandbResultLogging.NOTHING in self.prediction_logging_type:
                return False
        else:
            if WandbResultLogging.NOTHING in self.validation_test_logging_type:
                return False
        
        return True
