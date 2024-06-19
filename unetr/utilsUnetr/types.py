"""This module contains all kind of types to have a more
expressive and self-explanatory code with Python typing system.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


LabelColors = Dict[int, Tuple[int, int, int]]
LabelNames = Dict[int, str] # Names of the labels in the dict format.


class ActionType(Enum):
    """Enumerator of the different available actions inside the app.
    
    Attributes:
        TRAINING: Used to flag when we are training a model.
        VALIDATION: Used to flag when we are validating a model.
        TESTING: Used to flag when we are testing a model.
        PREDICTION: Used to flag when we are predicting with a model.
    """
    
    TRAINING = "train"
    VALIDATION = "val"
    TESTING = "test"
    PREDICTION = "pred"


class Metrics(Enum):
    """Enumerator which lists the different available metrics to log, verify, ...
    
    Attributes:
        DICE: Used to use the Dice Score Coefficient as a metric for the model.
        HAUSDORFF_DISTANCE_95: Used to use the Hausdorff Distance (95th percentile)
            as a metric for the model.
    
    See also:
        monai.metrics.DiceMetric: Dice metric associated.
        monai.metrics.HausdorffDistanceMetric: Hausdorff Distance metric associated.
    """
    
    DICE = "dice"
    HAUSDORFF_DISTANCE_95 = "hd95"


class PredictionSavingType(Enum):
    """Enumerator which lists the different available type of saving predicted images.
    
    Attributes:
        ALL: Save all the predicted image.
        RANDOM: Only save one random slice from the predicted image.
        AS_NIFTI: Save the image in a Nifti file format.
        NOTHING: Save no images or slices from a predicted image.
    """

    ALL = "all"
    RANDOM = "random"
    AS_NIFTI = "nifti_matrix"
    NOTHING = "nothing"

    def __init__(self, slice_idx: Union[str, int] = None) -> None:
        """
        Arguments:
            slice_idx: Can be a specified slice or a default value of the enumerator available values (optional).
        """
        self.slice_idx = None
        if isinstance(slice_idx, int) and slice_idx >= 0:
            self.slice_idx = slice_idx

    @staticmethod
    def verify(values: List["PredictionSavingType"]) -> None:
        """Verifies a list of values.
        
        You can't specify, NOTHING or ALL with another value for saving images.
        You can activate or not the Nifti format with any other saving type
        (excluding NOTHING).
        
        Arguments:
            values: List of saving types to check the consistancy of the values.
        
        Raise:
            ValueError: When a value can't be used with another one.
        """
        if len(values) == 0:
            raise ValueError("you must have to specify a saving type")

        if len(values) > 1 and PredictionSavingType.NOTHING in values:
            raise ValueError("you can't have nothing to save and save another thing")

        if PredictionSavingType.ALL in values and PredictionSavingType.RANDOM in values:
            raise ValueError("cannot have 'PredictionSavingType.ALL' and 'PredictionSavingType.RANDOM' " \
                                 "in the same list values. please choose only one of these")
        
        values_with_slice_idx = PredictionSavingType.slices_to_save(values)
        if len(values_with_slice_idx) != 0 and (PredictionSavingType.ALL in values or PredictionSavingType.RANDOM in values):
            raise ValueError("cannot have 'PredictionSavingType.ALL' or 'PredictionSavingType.RANDOM' with a specific slice to save" \
                                 " in the same list values. please choose only one of these")
    
    @staticmethod
    def slices_to_save(values: List["PredictionSavingType"]) -> List[int]:
        """Gets the slices to save. This method is interesting only if there are some specific
        slices to save.

        Arguments:
            values: List of saving types to retrieve slice indices for a future saving.

        Returns:
            values_with_slice_idx: Prediction saving types that there have integer slice index as attribute.
        """
        values_with_slice_idx = []

        for value in values:
            if value.slice_idx is not None:
                values_with_slice_idx.append(value)
        
        return values_with_slice_idx
