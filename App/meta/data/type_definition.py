import abc
from enum import Enum
import os
from typing import List, NamedTuple, Optional, Sequence, Tuple, TypedDict, Union

from monai.transforms.transform import Transform
from monai.transforms import LoadImaged
import numpy as np
import pandas as pd
import torch

class MetaReader(str, Enum):
    """Class made to know what type of data was use and what the user need to put for those data"""
    DICOM = "dicom"
    NIFTI = "nifti"
    NIFTI_MULTIPLE_MASKS = "nifti_multiple_masks"


class MetaDictObject(TypedDict):
    image: Union[str, np.ndarray, torch.Tensor]
    label: Union[str, np.ndarray, torch.Tensor]

class MetaIntermediateItem(NamedTuple):
    """class with the itermediate item of the meta"""
    dict_object: Optional[Union[MetaDictObject, Sequence[MetaDictObject]]]
    patient_id: str
    has_meta: bool


class MetaFinalItem(NamedTuple):
    """class with the final item of the meta"""
    image: Optional[Union[np.ndarray, torch.Tensor]]
    label: Optional[Union[np.ndarray, torch.Tensor]]
    patient_id: str
    has_meta: bool


class MetaDatasetReaderAbstract(abc.ABC):
    """class made to read the meta dataset"""
    def __init__(self, base_dir: str, df: pd.DataFrame, transform: Transform) -> None:
        """ Initialise the class MetaDatasetReaderAbstract
        Args:
        base_dir: string with the path of the directory where we are
        df: the dataframe
        transform: the transforation that we want to do in the data

        Returns:
        nothing
        """
        super().__init__()
        self.base_dir = base_dir
        self.transform = transform
        self.df = df
    
    def __len__(self) -> int:
        """ to get the length of the dataframe
        Returns:
        len of df
        """
        return self.df.shape[0]

    def get_patient_ids(self) -> Tuple[pd.Series, pd.Series]:
        """ to get the identifiant of patient
        Returns:
            the patient id and if he have a meta or not
        """
        return self.df["id"], self.df["has_meta"]
    
    def get_patient_id(self, idx: int) -> str:
        """ verify if the index is inferior to length of the dataframe and if it's return the identifiant
        Args:
            idx: index of the patient that we want to return the identifiant
        Returns:
            identifiant of the wanted patient
        """
        if idx >= len(self):
            raise ValueError("idx too large")
        return self.df[["id"]].values[idx][0]
    
    def patient_has_meta(self, idx: int) -> bool:
        """ verify if the patient identifiant is inferior to the length of the dataframe 
        after return True if this patient have a meta, false if not

        Args:
            idx: wanted identifiant of a patient
        Returns:
            boolean, true if this patient have a meta false if not

        """
        if idx >= len(self):
            raise ValueError("idx too large")
        return self.df[["has_meta"]].values[idx][0]

    def patient_has_meta_from_id(self, patient_id: str) -> bool:
        """ to know if the wanted patient with the enter id have a meta

        Args:
            patient_id: string which is the identifiant of the wanted patient
        Returns:
            a boolean which say if this patient have or not a meta
        """
        return self.df[self.df["id"] == int(patient_id)][["has_meta"]].values[0][0]

    @abc.abstractmethod
    def load(self, patient_id: str) -> MetaDictObject:
        raise NotImplementedError()


class MetaDatasetNiftiReaderAbstract(MetaDatasetReaderAbstract):
    """ This class is use to read a nifti dataset"""
    def __init__(self, base_dir: str, df: pd.DataFrame) -> None:
        """ Initialise the class MetaDatasetNiftiReaderAbstract
        Args:
            base_dir: string with the path of the directory where we are
            df: the dataframe
        """
        transform = LoadImaged(reader=("NibabelReader", "NibabelReader"), keys=["image", "label"])
        
        super().__init__(base_dir, df, transform=transform)
        self.__filter_df()


    def load(self, patient_id: str) -> MetaDictObject:
        """ create an object with the image and label of the wanted patient
            first we get the paths of the image and label
            after we create a dictionnary with 
            if we dont have label:
                the dictionary is only for load the image
            else:
                the dictionary is a self transformation


        Args:
            patient_id: identifiant of the wanted patient
        Returns:
            the dictionary with the image and label
        """
        img_path, mask_path = self._get_paths(patient_id)
        
        dict_object = {
            "image": img_path,
            "label": mask_path,
        }

        if not os.path.exists(mask_path):
            dict_object = LoadImaged(reader="NibabelReader", keys=["image"])(dict_object)
        else:
            dict_object = self.transform(dict_object)

        return dict_object
    
    @abc.abstractmethod
    def _get_paths(self, patient_id: str) -> Tuple[str, str]:
        raise NotImplementedError()
    
    def __filter_df(self):
        """ this function is here to filter the dataframe information, we take only the patient wich are existing
        after we recreate the dataframe"""
        def patient_exists(patient_id: str) -> bool:
            """verify if this patient is existing with the verification of the existance of him image
            Args:
                patient_id: a string with the identifiant of the patient
            Returns:
                a boolean which say if this patient have an image or not
            """
            img_path, _ = self._get_paths(patient_id)
            return os.path.exists(img_path)
        
        patients = self.df[[patient_exists(patient_id) for patient_id in self.df["id"].values]]
        self.df = patients



class ModuleVersionError(Exception):
    """Raised when the version of a module is bad."""
    pass
