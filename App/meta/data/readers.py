from typing import Tuple

from monai.transforms import LoadImaged
import rt_utils
import torch

from meta.utilsMeta import get_monai_version
from .input_output import (
    NiftiFilesArchitectureFor1Mask,
    NiftiFilesArchitectureForMultipleMasks,
    DicomFilesArchitecture,
)
from .type_definition import (
    MetaDictObject,
    MetaDatasetReaderAbstract,
    MetaDatasetNiftiReaderAbstract,
    ModuleVersionError,
)


class MetaDatasetOnlyNiftiMetaReader(MetaDatasetNiftiReaderAbstract):
    def __init__(self, base_dir: str) -> None:
        """initialise the subclass of MetaDatasetNiftiReaderAbstract name MetaDatasetOnlyNiftiMetaReader
        get the patient id in the directory pass in argument and put it in a dataframe df
        initialise the superclass
        Args:
        base_dir: directory where we want to work"""
        df = NiftiFilesArchitectureFor1Mask.get_patient_ids(base_dir)
        super().__init__(base_dir, df)
    
    def _get_paths(self, patient_id: str) -> Tuple[str, str]:
        """get the path of the slices and meta of the patient
        Args:
        patient_id: identifiant of the wanted patient
        Returns:
            2 things, the path of patient slices and the path of patient meta"""
        return (
            NiftiFilesArchitectureFor1Mask.get_slices_paths(self.base_dir, patient_id),
            NiftiFilesArchitectureFor1Mask.get_meta_mask_path(self.base_dir, patient_id)
        )


class MetaDatasetMultipleMasksNiftiReader(MetaDatasetNiftiReaderAbstract):
    def __init__(self, base_dir: str) -> None:
        """initialise the class MetaDatasetMultipleMasksNiftiReader which is a subclass of MetaDatasetNiftiReaderAbstract
        get a string of patient identifiant which are in base_dir and put it on the dataframe df
        initialise the superclass with base_dir and dr
        Args:
        base_dir: directory where we want to work"""
        df = NiftiFilesArchitectureForMultipleMasks.get_patient_ids(base_dir)
        super().__init__(base_dir, df)
    
    def _get_paths(self, patient_id: str) -> Tuple[str, str]:
        """ get the path of patient slices and meta
        Args:
        patient_id: the identifiant of the wanted patient
        Returns:
        2 things, the path of patient slices and the path of patient meta"""
        return (
            NiftiFilesArchitectureForMultipleMasks.get_slices_paths(self.base_dir, patient_id),
            NiftiFilesArchitectureForMultipleMasks.get_meta_mask_path(self.base_dir, patient_id)
        )


class MetaDatasetDicomReader(MetaDatasetReaderAbstract):
    def __init__(self, base_dir: str) -> None:
        """initialise the class MetaDatasetDicomReader, a subclass of MetaDatasetReaderAbstract
        verify if the monai version is less than 1,0,0, if it's not, return an error
        put in a dataframe df the patient id which are in base_dir
        put in transform the dicom images return by the reader
        initialise the superclass
        Args:
        base_dir: directory where we want to work"""
        if get_monai_version() < (1, 0, 0):
            raise ModuleVersionError("you need to have at least the monai==1.0.0 installed on your computer to use this reader")
        df = DicomFilesArchitecture.get_patient_ids(base_dir)
        transform = LoadImaged(reader=("PydicomReader"), keys=["image"])

        super().__init__(base_dir, df, transform=transform)
    
    def load(self, patient_id: str) -> MetaDictObject:
        """load the wanted patient
        verify if the patient have a meta or not and return it in consequence
        Args:
        patient_id: identifiant of the wanted patient
        Returns:
        the images and mask of a patient with or without a meta"""
        if self.patient_has_meta_from_id(patient_id):
            return self.__load_patient_with_meta(self.__get_mask_path(patient_id))
        return self.__load_patient_without_meta(self.__get_dicom_serie_path(patient_id))
    
    def __load_patient_with_meta(self, dict_object: MetaDictObject) -> MetaDictObject:
        """permite to load all the patient which have meta
        we create the RTSTRUCT with patients images and labels
        we take in the RTSTRUCT mask name meta and put it in the place of the old label in dict_object
        we return the dictionary after transformation
        Args:
        dict_object: dictonary with patient images and meta
        Returns:
        the transfomation of the dict_object"""
        rt_struct = rt_utils.RTStructBuilder.create_from(dict_object["image"], dict_object["label"])
        dict_object["label"] = rt_struct.get_roi_mask_by_name("meta")

        return self.transform(dict_object)

    def __get_mask_path(self, patient_id: str) -> MetaDictObject:
        """we want to get for a specific a dictionary with images dicom and label
        we extract the dicom image of our patient
        we extract the mask 
        Args:
        patient_id: identifiant of the wanted patient
        Returns:
        a dictionary with the dicoms image in image and the mask in label"""
        # TODO: handle when more than one CT-scan are available (ex: 14-002)
        dicom_images = self.__get_dicom_serie_path(patient_id)
        mask_image = DicomFilesArchitecture.get_meta_mask_path(self.base_dir, patient_id)[0]
        return {"image": dicom_images, "label": mask_image}

    def __get_dicom_serie_path(self, patient_id: str) -> str:
        """to get the path of the dicom images of a specific patient
        Args:
        patient_id: identifiant of the wanted patient
        Returns:
        the path of the patient slices"""
        return DicomFilesArchitecture.get_slices_dir_path(self.base_dir, patient_id)[0]

    def __load_patient_without_meta(self, patient_scan_path: str) -> MetaDictObject:
        """load the patent who dont have meta
        we transform the image before put it in the dictionary return
        Args:
        patient_scan_path: path where we can get the scan of patient who dont have meta
        Returns:
        a dictionary with image and a black image for label"""
        dict_object = self.transform({"image": patient_scan_path, "label": None})
        dict_object["label"] = torch.zeros_like(dict_object["image"])
        return dict_object

