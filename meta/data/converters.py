import glob
import os
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

import dcmrtstruct2nii
from monai.transforms import LoadImage
import nibabel as nib
import numpy as np
import pandas as pd
import rt_utils
import tqdm

from .input_output import (
    DicomFilesArchitecture,
    NiftiFilesArchitectureFor1Mask,
    NiftiFilesArchitectureForMultipleMasks,
)


class NiftiMetaMaskExtractorFromDicomDataset:
    """class create for the extraction of all the mask in nifty from a dicom dataset"""
    def __init__(self, base_dir: str) -> None:
        """ init the class NiftiMetaMaskExtractorFromDicomDataset
        with base_dir pass in arguments and df take from an other class and made with the identifiant of all the dataset patient
        Args:
        base_dir: the path of the directory that we wanted to be in
        """
        self.base_dir = base_dir
        self.df = DicomFilesArchitecture.get_patient_ids(base_dir)

    def __get_meta_mask_path(self, patient_id: str) -> str:
        """we want to get the meta mask of a specific patient
        Args:
        paient_id: identifiant of the wanted patient
        Returns: None if we dont have meta mask and the meta mask if we have it"""
        paths = DicomFilesArchitecture.get_meta_mask_path(self.base_dir, patient_id)
        return None if len(paths) == 0 else paths[0]

    def __get_slices_path(self, patient_id: str) -> str:
        """get the directory path where this patient is
        Args: patient_id: indentifiant of the wanted patient
        """
        return DicomFilesArchitecture.get_slices_dir_path(self.base_dir, patient_id)[0]

    def __save_image_as_nifti(self, array: np.ndarray, file_output: str):
        """ save the image in the nifty format
        if the length of our array is 4, we take only the first part of it 
        Args: array: the image 
        file_output: where we want to save the nifty image
        """
        if len(array.shape) == 4:
            array = array[0]
    
        image = nib.Nifti1Image(array, np.eye(4))
        nib.save(image, file_output)
        
       
    def fusionMask(mask):
        if (type(mask)==list):
            premMask = mask[0]
            for autMask in mask[1:]:
                premMask = np.where(autMask > 0, autMask, premMask)
            return premMask
        else:
            return mask

    def __load_dicom_with_mask(self, slices_folder: str, mask_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """ load all the image who have a mask
        Args:
        slices_folder: string with the path of the wanted folder
        mask_file: the mask of the slices_folder images
        Returns:
        the images and masks
        """
        rt_struct = rt_utils.RTStructBuilder.create_from(slices_folder, mask_file)
        maskAllAlone=[]
        for i in rt_struct.get_roi_names():
            if "GTV" in i or "FRONTAL" in i or "GTC" in i or "Cerebelleux" in i or "PARIET" in i or "gtv" in i or "M1." in i or "M2." in i or "ANC" in i or "FRONTAL" in i or "Cavite" not in i or "cavite" not in i or "PTV" not in i or "ANCIEN" not in i or "PTC" not in i:
                maskAllAlone.append(rt_struct.get_roi_mask_by_name(i).astype(np.float32))
        mask = fusionMask(mask).transpose([2, 0, 1])
        img = LoadImage("PydicomReader", image_only=True, force=True)(slices_folder).numpy().transpose([2, 0, 1])
        img = np.rot90(img)
        img= np.flip(img, 0)
        return img, mask

    def __load_dicom_series_without_mask(self, slices_folder: str) -> Tuple[np.ndarray, np.ndarray]:
        """load all the image which have not masks
        Args: slices_folder: string with the path of the wanted folder
        Returns:
        the images and masks
        """
        img = LoadImage("PydicomReader", image_only=True, force=True)(slices_folder).numpy().transpose([2, 0, 1])
        img = np.rot90(img)
        img= np.flip(img, 0)
        mask = np.zeros_like(img)
        return img, mask

    def convert2nii(self, target_dir: str):
        """convert the dicom data to nifty
        we create all the needded directory in the target one
        parcour pbar and get for all passage the patient_id and the boolean which say if the patient have or not a meta
        get the slices path and the meta path where we have this patient and is mask
        if we have meta or mask, we load it with the function load_dicom_with_mask
        else, we load it with the function load_dicom_series_without_mask
        Finaly, we save the image and is mask to nifti
        Args: 
        target_dir: path of the target directory
        Raises: 
        the error with the patient_id
        """
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(NiftiFilesArchitectureFor1Mask.get_slices_dir_path(target_dir), exist_ok=True)
        os.makedirs(NiftiFilesArchitectureFor1Mask.get_meta_mask_dir_path(target_dir), exist_ok=True)

        pbar = tqdm.tqdm(self.df.itertuples(name=None), total=self.df.shape[0], leave=True)
        for row in pbar:
            pbar.set_description(f"Converting {row[1]}...")

            patient_id = row[1]
            has_meta = row[2]

            try:
                slices_folder = self.__get_slices_path(patient_id)
                mask_file = self.__get_meta_mask_path(patient_id)

                if has_meta or mask_file is not None:
                    img, mask = self.__load_dicom_with_mask(slices_folder, mask_file)
                else:
                    img, mask = self.__load_dicom_series_without_mask(slices_folder)
                self.__save_image_as_nifti(img, NiftiFilesArchitectureFor1Mask.get_slices_paths(target_dir, patient_id))
                self.__save_image_as_nifti(mask, NiftiFilesArchitectureFor1Mask.get_meta_mask_path(target_dir, patient_id))
            except Exception as e:
                print("Error with", patient_id)
                print(e)


class NiftiMasksExtractorFromDicomDataset:
    """this class permite to extract the label of patient in the extention nifti from a dicom dataset"""
    def __init__(self, base_dir: str) -> None:
        """ initialise the class NiftiMasksExtractorFromDicomDataset
        with the initialisation of the base_dir which is the directory where we want to get the data
        and the df which is a list of all patient in the base_dir
        Args:
        base_dir: directory source"""
        self.base_dir = base_dir
        self.df = DicomFilesArchitecture.get_patient_ids(base_dir)

    def __get_meta_mask_path(self, patient_id: str) -> str:
        """this function is made to get the meta of the patient past in argument if he have one
        Args:
        patient_id: identifiant of the wanted mask patient
        Returns:
        None if this patient dont have meta
        the path to get his meta else"""
        paths = DicomFilesArchitecture.get_meta_mask_path(self.base_dir, patient_id)
        return None if len(paths) == 0 else paths[0]
    
    def __get_other_masks_path(self, patient_id: str) -> str:
        """here we can get all the mask of a patiient past in argument which are not the meta mask
        Args:
        patient_id: identifiant of the wanted mask patient
        Returns:
        All the other mask than meta of the wanted patient"""
        return DicomFilesArchitecture.get_other_mask_paths(self.base_dir, patient_id)[0]

    def __get_slices_path(self, patient_id: str) -> str:
        """ This function return the path to a directory that contains slices for the patient pass in argument.
        Args:
        patient_id: identifiant of the wanted mask patient
        Returns:
        the slices of the given patient"""
        return DicomFilesArchitecture.get_slices_dir_path(self.base_dir, patient_id)[0]

    def convert2nii(self, target_dir: str):
        """Convert a dataset in dicom to nifti format
        At first it creates an empty list "all_roi_names"
        create the directory, if it is not existe, which the path is pass on argument of this function
        Set up a progress bar which is growing all the time that a id of the data was treated
        for each patient we:
            get all his mask
            get all slices
            get the path of the slices where we want to store the nifti data
            create if it's not exist the directory where we will put the nifti
            get the list off the names in the RTSTRUCT
            convert RTSTRUCT and the slices to nifti and store them
            if we have a meta for this patient, we convert it and store it
            we had the names contain in RTSTRUCT in "all_roi_names"
            to finish, we create a csv file with all the names of all the RTSTRUCT which are not duplicate
        Args:
        target_dir: directory where we want to put the nifti data
        Raises: If we have an error when we extract the data of the RTSTRUCT, we print that we had an issus with the patient "identifiant" """
        all_roi_names = []
        os.makedirs(target_dir, exist_ok=True)

        pbar = tqdm.tqdm(self.df["id"].values)
        for patient_id in pbar:
            pbar.set_description(f"Processing {patient_id}")

            masks_path = self.__get_other_masks_path(patient_id)
            meta_mask_path = self.__get_meta_mask_path(patient_id)
            dicom_serie_path = self.__get_slices_path(patient_id)
            outdir = NiftiFilesArchitectureForMultipleMasks.get_slices_dir_path(target_dir, patient_id)
            os.makedirs(outdir, exist_ok=True)

            try:
                roi_names = dcmrtstruct2nii.list_rt_structs(masks_path)
                dcmrtstruct2nii.dcmrtstruct2nii(masks_path, dicom_serie_path, outdir)
                if meta_mask_path is not None:
                    dcmrtstruct2nii.dcmrtstruct2nii(meta_mask_path, dicom_serie_path, outdir, convert_original_dicom=False)

                all_roi_names.extend(roi_names)
            except Exception:
                print(f"RTSTRUCT extraction error for patient with {patient_id} as id")

        df = pd.DataFrame(data=all_roi_names, columns=["roi_names"])
        df = df.drop_duplicates()
        df = df.sort_values("roi_names")
        df = df.reset_index(drop=True)
        df.to_csv(
            NiftiFilesArchitectureForMultipleMasks.get_csv_path(target_dir, NiftiFilesArchitectureForMultipleMasks.ALL_ROI_NAMES_CSV_FILENAME),
            index=False
        )
        self.df.to_csv(os.path.join(self.base_dir, NiftiFilesArchitectureForMultipleMasks.PATIENTS_CSV_FILENAME))

    @staticmethod
    def generate_mapping_csv(dir: str):
        """renames names of the RTSTRUCT to assamble a part of them 
        Read the csv file in the given directory
        rename the collomns
        save the csv file
        Args:
        dir:directory which contain a csv file"""
        df = pd.read_csv(NiftiFilesArchitectureForMultipleMasks.get_csv_path(dir, NiftiFilesArchitectureForMultipleMasks.ALL_ROI_NAMES_CSV_FILENAME))
        df = df.rename({"roi_names", "old_roi_names"}, axis=1)
        df["new_roi_names"] = df["old_roi_names"]
        df.to_csv(NiftiFilesArchitectureForMultipleMasks.get_csv_path(dir, NiftiFilesArchitectureForMultipleMasks.MAPPER_ROI_NAMES_CSV_FILENAME), index=False)
    
    @staticmethod
    def generate_labels(dir: str) -> pd.DataFrame:
        """generate label for csv file
        read a csv file and put it in  "mapping_df"
        remove all the duplicate rows and keep the first occurence of it
        reset the dataset to have a new index
        remove the old_roi_names collomn
        rename roi_name to new_roi_names
        creation of a new colomn, label which contain a list of integer start to 0 and finish when we have parcour all the names
        rearangement of "mapping_df" with only tow colomn, roi_name and label
        save the csv
        Args:
        dir: directory wich containe a csv file
        Returns:
        the new mapping for the csv file"""
        mapping_df = pd.read_csv(os.path.join(dir, NiftiFilesArchitectureForMultipleMasks.MAPPER_ROI_NAMES_CSV_FILENAME))
        mapping_df = mapping_df.drop_duplicates(subset="new_roi_names", keep="first")
        mapping_df = mapping_df.reset_index(drop=True)
        mapping_df = mapping_df.drop(columns="old_roi_names")
        mapping_df = mapping_df.rename({"new_roi_names", "roi_name"}, axis=1)
        mapping_df["label"] = np.arange(len(mapping_df.index))

        mapping_df = mapping_df[["roi_name", "label"]]
        mapping_df.to_csv(NiftiFilesArchitectureForMultipleMasks.get_csv_path(dir, NiftiFilesArchitectureForMultipleMasks.LABELS_CSV_FILENAME), index=False)
        return mapping_df
    
    @classmethod
    def rename_files_from_csv(cls, dir: str):
        """renames nifti files with csv files mapping
        put the csv file in "mapper"
        renames the multiple mask of patient with the names in the csv files
        Args:
        dir: directory with the csv files
        Returns:
        the rename for files"""
        mapper = pd.read_csv(NiftiFilesArchitectureForMultipleMasks.get_csv_path(dir, NiftiFilesArchitectureForMultipleMasks.MAPPER_ROI_NAMES_CSV_FILENAME))
        return cls.rename_files(dir, mapper)
    
    @staticmethod
    def rename_files(dir: str, mapping_df: pd.DataFrame):
        """renames the nifti files in dir
        get a list of patient directory in "dir"
        create a progress bar which progress when patient_folders treatment progress to
        for each patient directory, the bar progress
            for each row, we construct the path to the nifti file
            if we have a path which is existing:
                we rename the file with the corresponding label
        Args:
        dir: directory source
        mapping_df: dataframe with roi name and label"""
        patient_folders = NiftiFilesArchitectureForMultipleMasks.patient_folders_only(dir)

        pbar = tqdm.tqdm(patient_folders)
        for folder in pbar:
            pbar.set_description(f"Processing {folder}...")

            for row in mapping_df.itertuples(name=None):
                path = os.path.join(folder, NiftiFilesArchitectureForMultipleMasks.mask_filename(row[1]))
                if os.path.exists(path):
                    os.rename(path, os.path.join(folder, NiftiFilesArchitectureForMultipleMasks.mask_filename(row[2])))

    @staticmethod
    def combine_masks(dir: str, patient_id: str, labels_df: pd.DataFrame):
        """combine the mask of a specific patient
        first, we search the files in the patient nifti directory and verify if we had more than one nifti file in in
        if we have, we store the path of that patient on "patient_folder"
        to finish, we create an instance of loadImage with the reader set with the nifti reader
        Args:
        patient_id: identifiant of the wanted patient
        labels_df: labels of the dataFrames
        Returns: if we have no more file than one, we return None"""
        if len(glob.glob(os.path.join(NiftiFilesArchitectureForMultipleMasks.get_slices_dir_path(dir, patient_id), "*"))) <= 1:
            return None

        patient_folder = glob.glob(NiftiFilesArchitectureForMultipleMasks.get_slices_dir_path(dir, patient_id))
        loader = LoadImage(reader=("NibabelReader"), image_only=True)

        def step(first_img, filename, label):
            """load one nifti file and normalise it and add it to a 3D array
            load the image data from the nifti file and normalise it to have a boolean image
            if the first image is None:
                create a 3D array with only zeros with the same shape as the image in the nifti file
            else, the label part of the first image is equal to the image in the nifti file    
            Args:
            first_img: the first image
            filename: the file name
            label: the label
            Returns:
            the first image
            Raises:     
            See also:     
            Notes:
            References:   
            Examples:"""
            img = (loader(filename) / 255).astype(bool)

            if first_img is None:
                first_img = np.zeros((len(labels_df.index), *img.shape), dtype=bool)
            else:
                first_img[label, :] = img
            
            return first_img

        img = None
        pbar = tqdm.tqdm(labels_df.itertuples(name=None), total=labels_df.shape[0], leave=True)
        for row in pbar:
            pbar.set_description(f"Processing {row[1]}...")

            path = os.path.join(patient_folder, NiftiFilesArchitectureForMultipleMasks.mask_filename(row[1]))
            img = step(img, path, row[2])
        
        nifti_img = nib.Nifti1Image(img, np.eye(4))
        nib.save(nifti_img, NiftiFilesArchitectureForMultipleMasks.get_final_mask_path(dir, patient_id))

    @classmethod
    def combine_masks_all_patient(cls, dir: str):
        """combines mask of all the patient
        read the csv file in the directory source
        get a list of the patient directory in dir
        set up a progress bar difine one the patient_folder treatment
        for all the patient, we extract them identifiant and combine patient id and label of the data frame
        Args:
        dir: directory sources"""
        labels_df = pd.read_csv(NiftiFilesArchitectureForMultipleMasks.get_csv_path(dir, NiftiFilesArchitectureForMultipleMasks.LABELS_CSV_FILENAME))
        patient_folders = NiftiFilesArchitectureForMultipleMasks.patient_folders_only(dir)

        pbar = tqdm.tqdm(patient_folders, leave=True)
        for folder in pbar:
            pbar.set_description(f"Processing {folder}...")

            patient_id = folder.split(os.sep)[-1]
            cls.combine_masks(dir, patient_id, labels_df)
