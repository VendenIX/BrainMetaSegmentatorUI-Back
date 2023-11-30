from datetime import datetime
from enum import Enum
import json
import os
from typing import List, Optional, Tuple, TypedDict, Union

from monai.transforms.transform import Transform
import numpy as np
import torch
import torch.utils.data as data

from meta.random_ import create_generator
from .dataset import MetaDataset, MetaSubset
from .type_definition import MetaReader


class DatasetType(str, Enum):
    """This class represents the different type of datasets that we will use in the current learning

    Args:
        TRAINING: use to use something for the training (recuperate the good things for the training)
        VALIDATION: use to use something for the valisation (recuperate the good things for the validation)
        TESTING: use to use something for the testing (recuperate the good things for the testing)
    See also:
            
    Notes:

    References:
            
    Examples:

    """
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class DatasetDictIndices(TypedDict):
    """This class represents the different type of datasets with them possible type therforce, they can be integer, numpy array or pytorch tensor.

    Args:
        DatasetType.TRAINING: is the dictionnaire of the datasetfor training
        DatasetType.VALIDATION: is the dictionnaire of the datasetfor validation
        DatasetType.TESTING: is the dictionnaire of the datasetfor testing
    See also:
            
    Notes:

    References:
            
    Examples:

    """
    DatasetType.TRAINING: Union[List[int], np.ndarray, torch.Tensor]
    DatasetType.VALIDATION: Union[List[int], np.ndarray, torch.Tensor]
    DatasetType.TESTING: Union[List[int], np.ndarray, torch.Tensor]


class MetaDatalist:
    """Here we have a class which is creating the dataset with its function.
    Args:

    See also:
            
    Notes:

    References:
            
    Examples:

    """
    def __init__(self, data_dir: str, reader: MetaReader,
                 train_transform: Optional[Transform] = None, 
                 val_transform: Optional[Transform] = None, 
                 test_transform: Optional[Transform] = None, 
                 lengths: Tuple[float, float, float] = (0.7, 0.1, 0.2), 
                 generator_seed: Optional[int] = None, device: torch.device = None,
                 dict_of_indices: Optional[DatasetDictIndices] = None,
                 deactivate_shuffle: bool = False,
                 dtype = torch.float32) -> None:
        """This function is here to define the class MetaDatalist and its common variable.
            first, we attribut arguments to variables.
            second, if we have a dict_of_indices, we use it for the creation of the three part of the initial dataset
                    else we make a random split with the number put on the variable lengths to split the dataset in three part with this value par part


        Args:
            data_dir: string with the data in the source directory 
            reader: an instance of a class which is created to read the meta file.
            train_transform: the transformation that the user want to apply on the training data
            val_transform: the transformation that the user want to apply on the valuation data
            test_transform: the transformation that the user want to apply on the test data
            lengths: lengths of tree part with a total of one for split the data in three part (training, validation, testing)
            generator_seed: an int for the creation of a seed
            dict_of_indices: an instance of a class DatasetDictIndices with the indices of what we want to put in every dataset part
            deactivate_shuffle: a boolean to know if we want a random take dataset or a dataset take in the order of it given on the source directory
            dtype: PyTorch output dtype for the data.

        Returns:

        Raises:
                
        See also:
                
        Notes:

        References:
                
        Examples:
        """

        assert sum(lengths) == 1.0

        dataset = self.__create_dataset(data_dir, reader, device=device, dtype=dtype)
        self._patient_ids, self._has_meta = dataset.get_patient_ids()
        self.reader = reader
        print("le dict of indice", dict_of_indices)
        if dict_of_indices is not None:
            self.__train_dataset = MetaSubset(dataset, dict_of_indices[DatasetType.TRAINING])
            self.__validation_dataset = MetaSubset(dataset, dict_of_indices[DatasetType.VALIDATION])
            self.__test_dataset = MetaSubset(dataset, dict_of_indices[DatasetType.TESTING])
        else:
            self.__train_dataset, self.__validation_dataset, self.__test_dataset = \
                self.__random_split(dataset, lengths,
                    create_generator(generator_seed) if generator_seed is not None else None,
                    deactivate_shuffle=deactivate_shuffle,
                )
        print("les datasets",self.__train_dataset, self.__validation_dataset, self.__test_dataset)
        self.__train_dataset.set_transform(train_transform)
        self.__validation_dataset.set_transform(val_transform)
        if lengths[-1] < 1.0: # to check the test metrics
            self.__test_dataset.set_transform(val_transform)
        else: # just for prediction
            self.__test_dataset.set_transform(test_transform)
        
    def __create_dataset(self, data_dir: str, reader: MetaReader,
                         device: torch.device = None, dtype = torch.float32) -> MetaDataset:
        """This function return a dataset which will be use on the learning
        Args:
            data_dir: string with the data in the source directory 
            reader: an instance of a class which is created to read the meta file.
            device: the device for the tensor that is return in this function
            dtype: PyTorch output dtype for the data.
        Returns:
            return an instance of the class MetaDataset 
        Raises:
                
        See also:
            meta.data.dataset.MetaDataset
        Notes:

        References:
                
        Examples:
        """
        return MetaDataset(data_dir, reader, device=device, dtype=dtype)
    
    def __random_split(self, dataset: MetaDataset, lengths: Tuple[float, float, float], 
                generator: Optional[int] = None, deactivate_shuffle: bool = False) -> Tuple[MetaSubset, MetaSubset, MetaSubset]:

        """This function split the data in three dataset with the length indicate in the variable lengths. The data have an random repartition.
            
            First, we get bac the length for every futur dataset
            second, if we dont desactivate the shuffle:
                                update the variable lengths with the calculated values
                                if we have a seed:
                                            create the three dataset with splitting "dataset" according to the lengths with the seed "generator"
                                else:
                                            create the three dataset with splitting "dataset" according to the lengths
                                return: three instance of MetaSubset  with the connected subset.
                    without the shuffle:
                                we split "dataset" in its original order and create three instance of this.

        Args:
            dataset: an instance of the class MetaDataset with the data that we want to spleet in three part
            lengths: length of the futur dataset first for training, after validation and at final testing
            generator: the possible seed for the random split
            deactivate_shuffle: boolean that say if we shuffle the data or not

        Returns:
            the three dataset split according to "lengths"
        """
        length = len(dataset)
        train_length = int(length * lengths[0])
        val_length = int(length * lengths[1])
        test_length = length - train_length - val_length

        if not deactivate_shuffle:
            lengths = (train_length, val_length, test_length)
            if generator is not None:
                train_dataset, validation_dataset, test_dataset = \
                        data.random_split(dataset, lengths, generator)
            else:
                train_dataset, validation_dataset, test_dataset = \
                        data.random_split(dataset, lengths)
            return (
                MetaSubset.from_subset(train_dataset),
                MetaSubset.from_subset(validation_dataset),
                MetaSubset.from_subset(test_dataset)
            )
        
        train_dataset = MetaSubset(dataset, indices=range(0, train_length))
        validation_dataset = MetaSubset(dataset, indices=range(train_length, train_length + val_length))
        test_dataset = MetaSubset(dataset, indices=range(train_length + val_length, length))
        return train_dataset, validation_dataset, test_dataset

    @classmethod
    def from_json(cls, json_file: str, data_dir: str, train_transform: Optional[Transform] = None,
                  val_transform: Optional[Transform] = None, test_transform: Optional[Transform] = None, 
                  device: torch.device = None, dtype = torch.float32) -> "MetaDatalist":
        """Here we recover a list of the datas which were in a json file
        We open the json and put it in "file" and, after, load the variable "file" and put it in a dictionary

        Args:
            cls: our actual classes
            json_file: the json file use to extract data
            data_dir: a path to a directory 
            train_transform: transformation on the train data
            val_transform: transformation on the valuation data
            test_transform: transformation on the test data
            device: the device we want to use
            dtype: PyTorch output dtype for the data.
        Returns:
            a instance of MetaDatalist
        Raises:
                
        See also:
                
        Notes:

        References:
                
        Examples:
        """

        with open(json_file, "r") as file:
            dict_object = json.load(file)
        return cls(data_dir, dict_object["reader"], train_transform=train_transform, val_transform=val_transform,
                   test_transform=test_transform, device=device, dict_of_indices=dict_object["dataset"], dtype=dtype)

    def to_json(self, json_file: str) -> None:

        """Here we had dataset to a json file
            we create an object "dict_object" where we put the patient from  three dataset Training, Validation and Testing
            after, we open "json_file" and add to it the dict_object
        Args:
            json_file: a path for getted a json file
        Returns:
            
        Raises:
                
        See also:
                
        Notes:

        References:
                
        Examples:
        """
        dict_object = {
            "name": "Meta dataset",
            "created": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
            "reader": self.reader,
            "dataset": {
                DatasetType.TRAINING: [self.__train_dataset.dataset.get_patient_id(idx) for idx in self.__train_dataset.indices],
                DatasetType.VALIDATION: [self.__validation_dataset.dataset.get_patient_id(idx) for idx in self.__validation_dataset.indices],
                DatasetType.TESTING: [self.__test_dataset.dataset.get_patient_id(idx) for idx in self.__test_dataset.indices],
            },
        }
        with open(json_file, "w") as file:
            json.dump(dict_object, file)
    
    def get_subset(self, dataset_type: DatasetType) -> MetaSubset:
        """This is created to get back the wanted Datasubset.
        We verify what want the user and give the wanted variable in result
        Args:
            dataset_type: the type of the dataset that the user want
        Returns:
            a MetaSubset
        Raises:
              if the dataset_type enter is not like the function want it, we explain to the user that we want a more precise "dataset_type"  
        See also:
                
        Notes:

        References:
                
        Examples:
        """
        if dataset_type == DatasetType.TRAINING:
            return self.__train_dataset
        
        if dataset_type == DatasetType.VALIDATION:
            return self.__validation_dataset
        
        if dataset_type == DatasetType.TESTING:
            return self.__test_dataset
        
        raise ValueError("the specified dataset_type is invalid")

    def get_patient_ids(self, dataset_type: DatasetType = None) -> List[str]:
        """This is created to get back the patient ids.
        if the subset is not specify, we return all the patient id.
        else, we return all the patient id wich are in the wanted subset
        Args:
            dataset_type: the type of the dataset that the user want to return the patient id
        Returns:
            a list of string with the patient id
        Raises:
              
        See also:
                
        Notes:

        References:
                
        Examples:
        """
        if dataset_type is None:
            return self._patient_ids.tolist()
        
        subset = self.get_subset(dataset_type)
        return [subset.get_patient_id(idx) for idx in subset.indices]

    def get_ids_patient_with_meta(self, dataset_type: DatasetType = None) -> List[str]:
        """This is created to get back the patient ids who have meta on them data.
        if the subset is not specify, we return all the patient id with meta.
        else, we return all the patient id with meta wich are in the wanted subset
        Args:
            dataset_type: the type of the dataset that the user want to return the patient id with meta
        Returns:
            a list of string with the patient id with meta
        Raises:
              
        See also:
                
        Notes:

        References:
                
        Examples:
        """
        if dataset_type is None:
            return [self._patient_ids[i] for i in range(len(self._patient_ids)) if self._has_meta[i]]
        
        subset = self.get_subset(dataset_type)
        return [subset.get_patient_id(idx) for idx in subset.indices if subset.patient_has_meta(idx)]

    def get_ids_patient_without_meta(self, dataset_type: DatasetType = None) -> List[str]:
        """This is created to get back the patient ids without meta.
        if the subset is not specify, we return all the patient id without meta.
        else, we return all the patient id without meta wich are in the wanted subset
        Args:
            dataset_type: the type of the dataset that the user want to return the patient id without meta
        Returns:
            a list of string with the patient id without meta
        Raises:
              
        See also:
                
        Notes:

        References:
                
        Examples:
        """
        if dataset_type is None:
            return [self._patient_ids[i] for i in range(len(self._patient_ids)) if not self._has_meta[i]]
        
        subset = self.get_subset(dataset_type)
        return [subset.get_patient_id(idx) for idx in subset.indices if not subset.patient_has_meta(idx)]
