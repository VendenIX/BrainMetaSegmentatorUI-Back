import abc
import glob
import os
from typing import Iterable, List, Union

from monai.transforms import LoadImage
import pandas as pd
import inspect



class AbstractFilesArchitecture(abc.ABC):
    @abc.abstractclassmethod
    def patient_has_data(cls, *args) -> bool:
        """we verify if the patient have data
        do nohing if this methode doas not exist   """
        raise NotImplementedError()
    
    @abc.abstractclassmethod
    def patient_has_meta(cls, *args) -> bool:
        """we verify if the patient have a meta
        Raises:do nohing if this methode doas not exist    """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_meta_mask_dir_path(cls, *args) -> Union[str, List[str]]:
        """get the path to have the directory where are metas
        Raises: do nohing if this methode doas not exist"""
        raise NotImplementedError()
    
    @abc.abstractclassmethod
    def get_meta_mask_path(cls, *args) -> Union[str, List[str]]:
        """path to get the meta mask
        Raises:do nohing if this methode doas not exist  """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_other_mask_dir_path(cls, *args) -> Union[str, List[str]]:
        """get the path of the directory where are masks which are not meta
        Raises:do nohing if this methode doas not exist     """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_other_mask_paths(cls, *args) -> Union[str, List[str]]:
        """get the path to get the mask without meta
        Raises:   do nohing if this methode doas not exist  """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_slices_dir_path(cls, *args) -> Union[str, List[str]]:
        """get the path where are the patient slices
        Returns:
        the path
        Raises:    NotImplementedError() """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_slices_paths(cls, *args) -> Union[str, List[str]]:
        """get the patient slices
        Returns:
        the path
        Raises:  NotImplementedError()   """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_patient_ids(cls, base_dir: str) -> pd.DataFrame:
        """get the patient id
        Returns:
        the patient id
        Raises:  NotImplementedError()   """
        raise NotImplementedError()


class NiftiFilesArchitectureFor1Mask(AbstractFilesArchitecture):
    PATIENTS_CSV_FILENAME = "patients.csv"

    @classmethod
    def patient_has_data(cls, base_dir: str, patient_id: str) -> bool:
        """say if this patient have data
        get the data frame in bas_dir
        verify if there are something in the wanted part of the dataframe with the good id
        base_dir:directory where we want to work
        patient_id:identifiant of the wanted patient
        True if the patient have data"""
        df = cls.get_patient_ids(base_dir)
        return len(df[df["id"] == patient_id]) == 1
    
    @classmethod
    def patient_has_meta(cls, base_dir: str, patient_id: str) -> bool:
        """say if this patient have meta
        get the data frame in bas_dir
        verify if there are something in the wanted part of the dataframe with the good id
        Args:
        base_dir:directory where we want to work
        patient_id:identifiant of the wanted patient
        Returns:
        True if the patient have meta"""
        df = cls.get_patient_ids(base_dir)
        return df[df["id"] == patient_id][["has_meta"]].values[0][0]

    @classmethod
    def get_meta_mask_dir_path(cls, base_dir: str) -> str:
        """get the path of the directory where are the metas
        Args:
        base_dir:directory where we want to work
        Returns:
        the path"""
        return os.path.join(base_dir)

    @classmethod
    def get_meta_mask_path(cls, base_dir: str, patient_id: str) -> str:
        """get the path of the metas
        Args:
        base_dir:directory where we want to work
        Returns:
        the path"""
        return os.path.join(cls.get_meta_mask_dir_path(base_dir +"/"+ str(patient_id) +"/"), "mask_meta.nii.gz")
    
    @classmethod
    def get_slices_dir_path(cls, base_dir: str) -> str:
        """get the path of the directory where are the patients slices
        Args:
        base_dir:directory where we want to work
        Returns:
        the path"""
        return os.path.join(base_dir)

    @classmethod
    def get_slices_paths(cls, base_dir: str, patient_id: str) -> str:
        """get the path of the patient slices
        Args:
        base_dir:directory where we want to work
        Returns:
        the path"""
        return os.path.join(cls.get_slices_dir_path(base_dir +"/"+ str(patient_id) +"/"), "image.nii.gz")
    
    @classmethod
    def get_patient_ids(cls, base_dir: str) -> pd.DataFrame:
        """get the id of all patient in the wanted directory
        get the path of the csv file
        verify that the precedent path is existing
        if it is, return the readed csv
        else,get the patient id in the base dir with parcouring all the patient slices directory
        create a data frame with in the columns id all patient id
        load the images
        put in patient have meta all the path for meta for all patients
        put in the dataframe with hase_meta, the patient_has_meta variable
        transform the dataframe to a csv file
        Args:
        base_dir:directory where we want to work
        Returns:
        a dataframe with all the data"""
        path = os.path.join(base_dir, cls.PATIENTS_CSV_FILENAME)
        if os.path.exists(path):
            return pd.read_csv(path)
        patient_ids = [path.split(os.sep)[-1] for path in os.listdir(cls.get_slices_dir_path(base_dir))]
        df = pd.DataFrame(patient_ids, columns=["id"])
        loader = LoadImage(reader="NibabelReader")
        patient_has_meta = [
            loader(cls.get_meta_mask_path(base_dir, patient_id)).sum() == 0 
            for patient_id in patient_ids
        ]
        df["has_meta"] = patient_has_meta
        df.to_csv(path, index=False)
        return df


class NiftiFilesArchitectureForMultipleMasks(AbstractFilesArchitecture):
    PATIENTS_CSV_FILENAME = "patients.csv"
    ALL_ROI_NAMES_CSV_FILENAME = "all_roi_names.csv"
    MAPPER_ROI_NAMES_CSV_FILENAME = "mapped_roi_names.csv"
    LABELS_CSV_FILENAME = "labels.csv"

    @classmethod
    def patient_has_data(cls, base_dir: str, patient_id: str) -> bool:
        """verify if the patient have data
        verify if we have a patient slices directory and slices path
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:True if we have the path of slices directory and slices"""
        return (
            os.path.exists(cls.get_slices_dir_path(base_dir, patient_id)) 
            and 
            os.path.exists(cls.get_slices_paths(base_dir, patient_id))
        )
    
    @classmethod
    def patient_has_meta(cls, base_dir: str, patient_id: str) -> bool:
        """verify if the patient has meta
        get the meta path
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:True if we have a path for meta"""
        path = cls.get_meta_mask_path(base_dir, patient_id)
        return os.path.exists(path)

    @classmethod
    def get_meta_mask_dir_path(cls, base_dir: str, patient_id: str) -> str:
        """get the directory where are meta of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return os.path.join(base_dir, str(patient_id))
    
    @classmethod
    def get_meta_mask_path(cls, base_dir: str, patient_id: str) -> str:
        """get the path of meta of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return cls.mask_filename_path(base_dir, patient_id, "meta")

    @classmethod
    def get_other_mask_dir_path(cls, base_dir: str, patient_id: str) -> str:
        """get the path of mask withou meta of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return os.path.join(base_dir, patient_id)

    @classmethod
    def get_other_mask_paths(cls, base_dir: str, patient_id: str) -> str:
        """get the path of mask without meta of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return set(glob.glob(os.path.join(cls.get_other_mask_dir_path(base_dir, patient_id), "*.nii.gz"))) \
                - set([cls.get_meta_mask_path(base_dir, patient_id), cls.get_slices_paths(base_dir, patient_id)])

    @classmethod
    def get_slices_dir_path(cls, base_dir: str, patient_id: str) -> str:
        """get the path of the directory of wanted patient slices
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return os.path.join(base_dir, patient_id)

    @classmethod
    def get_slices_paths(cls, base_dir: str, patient_id: str) -> str:
        """get the path of wanted patient slices
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return os.path.join(cls.get_slices_dir_path(base_dir, patient_id), "image.nii.gz")
    
    @classmethod
    def get_final_mask_path(cls, base_dir: str, patient_id: str) -> str:
        """get the path of final mask of the wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return os.path.join(cls.get_slices_dir_path(base_dir, patient_id), "final_mask.nii.gz")

    @classmethod
    def get_patient_ids(cls, base_dir: str) -> pd.DataFrame:
        """get the id of all patient in the wanted directory
        get the path of the csv file
        verify that the precedent path is existing
        if it is, return the readed csv
        else,get the patient id in the base dir with parcouring all the patient slices directory
        create a data frame with in the columns id all patient id
        load the images
        put in patient have meta all the path for meta for all patients
        put in the dataframe with hase_meta, the patient_has_meta variable
        transform the dataframe to a csv file
        Args:
        base_dir:directory where we want to work
        Returns:
        a dataframe with all the data"""
        path = os.path.join(base_dir, cls.PATIENTS_CSV_FILENAME)
        if os.path.exists(path):
            return pd.read_csv(path)

        patient_folders = cls.patient_folders_only(base_dir)
        
        patient_ids = [folder.split(os.sep)[-1] for folder in patient_folders]
        patient_ids = list(filter(lambda patient_id: cls.patient_has_data(base_dir, patient_id), patient_ids))
        df = pd.DataFrame(patient_ids, columns=["id"])

        patient_has_meta = [cls.patient_has_meta(base_dir, patient_id) for patient_id in patient_ids]
        df["has_meta"] = patient_has_meta

        df.to_csv(path, index=False)
        return df

    @classmethod
    def mask_filename_path(cls, base_dir: str, patient_id: str, mask_name: str) -> str:
        """get the path of mask of wanted patient with a specifique file name
        Args:
        base_dir: directory where we want to work
        patient_id:identifiant od the wanted patient 
        mask_name: name of the mask we want
        Returns:
        the path"""
        return os.path.join(cls.get_other_mask_dir_path(base_dir, patient_id), cls.mask_filename(mask_name))

    @classmethod
    def mask_filename(cls, mask_name: str) -> str:
        """get the mask with the wanted name
        Args:
        mask_name:name of the mask we want
        Returns:
        the name of the file with this mask"""
        return f"mask_{mask_name}.nii.gz"

    @classmethod
    def patient_folders_only(cls, base_dir: str) -> Iterable[str]:
        """get all the path of folders in base_dir
        Args:
        base_dir: directory where we want to work
        Returns: list of all the files in base_dir except .csv and .json."""
        return list(set(glob.glob(os.path.join(base_dir, "*"))) - set(glob.glob(os.path.join(base_dir, "*.csv"))) - set(glob.glob(os.path.join(base_dir, "*.json"))))

    @classmethod
    def get_csv_path(cls, base_dir: str, csv_file: str) -> str:
        """get the path of .csv file
        Args:
        csv_file: name of the wanted file
        base_dir: directory where we want to work
        Returns: the path"""
        return os.path.join(base_dir, csv_file)


class DicomFilesArchitecture(AbstractFilesArchitecture):
    _fileext = ".dcm"

    @classmethod
    def patient_has_data(cls, base_dir: str, patient_id: str) -> bool:
        """verify if the wanted patient has all the nessessary data
        verify if they are folder in the patient directory
        verify if they are CT folder in there
        verify if they are data in CT folder in ther
        Args:
        base_dir: directory where we want to work
        patient_id: identifiant of the wanted patient
        Returns:
        True if we have all the necessary file else, False"""
        # if no folder in patient directory
        if len(glob.glob(os.path.join(base_dir, patient_id, "*"))) == 0:
            return False
        
        # if no named folder "CT"
        if len(cls.get_slices_dir_path(base_dir, patient_id)) == 0:
            return False

        # if not data inside "CT" folder
        if len(cls.get_slices_paths(base_dir, patient_id)) == 0:
            return False

        return True

    @classmethod
    def patient_has_meta(cls, base_dir: str, patient_id: str) -> bool:
        """verify if the wanted patient has meta data
        verify if they are META folder in the patient directory
        verify if they are data in META folder in ther
        Args:
        base_dir: directory where we want to work
        patient_id: identifiant of the wanted patient
        Returns:
        True if we have meta files else, False"""
        # if no named folder "META"
        if len(cls.get_meta_mask_dir_path(base_dir, patient_id)) == 0:
            return False

        # if not data inside "META" folder
        if len(cls.get_meta_mask_path(base_dir, patient_id)) == 0:
            return False

        return True

    @classmethod
    def get_meta_mask_dir_path(cls, base_dir: str, patient_id: str) -> List[str]:
        """get the path of META of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return glob.glob(os.path.join(base_dir, patient_id, "META"))
    
    @classmethod
    def get_meta_mask_path(cls, base_dir: str, patient_id: str) -> List[str]:
        """get the path of meta in META of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return glob.glob(os.path.join(base_dir, patient_id, "META", cls._fileext))

    @classmethod
    def get_other_mask_dir_path(cls, base_dir: str, patient_id: str) -> List[str]:
        """get the path of RTSTRUCT of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return glob.glob(os.path.join(base_dir, patient_id, "RTSTRUCT*"))

    @classmethod
    def get_other_mask_paths(cls, base_dir: str, patient_id: str) -> List[str]:
        """get the path of other mask in RTSTRUCT of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return glob.glob(os.path.join(base_dir, patient_id, "RTSTRUCT*", cls._fileext))

    @classmethod
    def get_slices_dir_path(cls, base_dir: str, patient_id: str) -> List[str]:
        """get the path of RM of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the path"""
        return glob.glob(os.path.join(base_dir, patient_id, "RM"))

    @classmethod
    def get_slices_paths(cls, base_dir: str, patient_id: str) -> List[str]:
        """get the path of all the slices in RM of wanted patient
        Args:
        base_dir: the directory where we want to work
        patient_id: the identifiant of patient
        Returns:the paths"""
        return glob.glob(os.path.join(base_dir, patient_id, "RM", cls._fileext))

    @classmethod
    def get_patient_ids(cls, base_dir: str) -> pd.DataFrame:
        """get a dataframe of patient ids
        get the ids of all patient in base_dir
        create a list with all the ids of patient who has data (verify with the eponym function)
        create the dataframe with the list of patient with data
        say if the patient has meta or not in an other columns of the dataframe
        Args:
        base_dir: the directory where we want to work
        Returns:
        the dataframe"""
        patient_ids = [path.split(os.sep)[-1] for path in glob.glob(os.path.join(base_dir, "*"))]

        patient_ids = list(filter(lambda patient_id: cls.patient_has_data(base_dir, patient_id), patient_ids))
        df = pd.DataFrame(patient_ids, columns=["id"])

        patient_has_meta = [cls.patient_has_meta(base_dir, patient_id) for patient_id in patient_ids]
        df["has_meta"] = patient_has_meta

        return df
