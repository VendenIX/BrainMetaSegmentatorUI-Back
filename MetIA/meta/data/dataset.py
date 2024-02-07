import abc
from typing import List, Optional, Sequence, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

from monai.transforms import Compose, EnsureChannelFirstd, Invertd, ToTensord
from monai.transforms.transform import Transform
import pandas as pd
import torch
import torch.utils.data as data

from .readers import (
    MetaDatasetOnlyNiftiMetaReader,
    MetaDatasetMultipleMasksNiftiReader,
    MetaDatasetDicomReader,
)
from .type_definition import (
    MetaIntermediateItem,
    MetaReader,
    MetaDatasetReaderAbstract,
    MetaFinalItem,
)


class MetaDatasetAbstract(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Union[MetaFinalItem, List[MetaFinalItem]]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_patient_id(self, idx: int) -> str:
        raise NotImplementedError()
        
    @abc.abstractmethod
    def patient_has_meta(self, idx: int) -> bool:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def set_transform(self, transform: Transform) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_transform(self) -> Transform:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_inverse_transform(self, transform: Optional[Transform] = None) -> Transform:
        raise NotImplementedError()


class MetaDataset(MetaDatasetAbstract, data.Dataset):
    def __init__(self, data_dir: str, reader: MetaReader,
                 transform: Optional[Transform] = None, 
                 device: torch.device = None,
                 dtype = torch.float32) -> None:
        self.device = device
        self._reader = self.__get_reader(reader, data_dir)
        self.transform = transform
        self._endload_transform_with_label = Compose([
            EnsureChannelFirstd(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
        ])
        self._endload_transform_without_label = Compose([
            EnsureChannelFirstd(keys=["image"]),
            ToTensord(keys=["image"]),
        ])
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, idx: Union[int, str]) -> Union[MetaFinalItem, List[MetaFinalItem]]:
        return self.get_item(idx, transform=self.transform)

    def get_item(self, idx: Union[int, str], transform: Optional[Transform] = None) -> Union[MetaFinalItem, List[MetaFinalItem]]:
        meta_item = self.get_item_without_transform(idx)

        if meta_item.dict_object is None:
            return meta_item

        return self.apply_transform(meta_item, transform=transform)
    
    def apply_transform(self, meta_item: MetaIntermediateItem, transform: Optional[Transform] = None) -> Union[MetaFinalItem, List[MetaFinalItem]]:
        dict_object, patient_id, has_meta = meta_item

        if self.transform is not None or transform is not None:
            transform = transform or self.transform
            dict_object = transform(dict_object)
        
        return self.apply_end_transformation(MetaIntermediateItem(dict_object, patient_id, has_meta))
    
    def apply_end_transformation(self, meta_item: MetaIntermediateItem) -> Union[MetaFinalItem, List[MetaFinalItem]]:
        dict_object, patient_id, has_meta = meta_item

        if isinstance(dict_object, list):
            output = [None]*len(dict_object)
            for ii in range(len(dict_object)):
                output[ii] = MetaFinalItem(
                    dict_object[ii]["image"],
                    dict_object[ii]["label"],
                    patient_id,
                    has_meta
                )
            return output
        
        return MetaFinalItem(
            dict_object["image"],
            dict_object["label"],
            patient_id,
            has_meta
        )
    
    def get_item_without_transform(self, idx: Union[int, str]) -> MetaIntermediateItem:
        if isinstance(idx, int):
            patient_id = self.get_patient_id(idx)
        else:
            patient_id = idx
        
        try:
            dict_object = self._reader.load(patient_id)
        except Exception:
            return MetaIntermediateItem(None, patient_id, False)

        # finish to load imgs
        if isinstance(dict_object["label"], str):
            dict_object = self._endload_transform_without_label(dict_object)
            dict_object["label"] = torch.zeros_like(dict_object["image"])
        else:
            dict_object = self._endload_transform_with_label(dict_object)

        dict_object["image"] = dict_object["image"].to(dtype=self.dtype)
        dict_object["label"] = dict_object["label"].to(dtype=self.dtype)

        # specific transforms
        if len(dict_object["image"].shape) == 3:
            dict_object["image"] = dict_object["image"][None, ...]
            dict_object["label"] = dict_object["label"][None, ...]
        dict_object["image"] = dict_object["image"].permute(0, 2, 1, 3).flip(3)
        dict_object["label"] = dict_object["label"].permute(0, 2, 1, 3).flip(3)

        if dict_object["label"].max() > 1.0:
            dict_object["label"] /= dict_object["label"].max()

        return MetaIntermediateItem(
            dict_object,
            patient_id,
            self._reader.patient_has_meta_from_id(patient_id)
        )
    
    def __get_reader(self, reader: MetaReader, data_dir: str) -> MetaDatasetReaderAbstract:
        if reader == MetaReader.DICOM:
            return MetaDatasetDicomReader(data_dir)
        elif reader == MetaReader.NIFTI:
            return MetaDatasetOnlyNiftiMetaReader(data_dir)
        elif reader == MetaReader.NIFTI_MULTIPLE_MASKS:
            return MetaDatasetMultipleMasksNiftiReader(data_dir)
        
        raise ValueError(f"the '{reader}' reader doesn't exist")

    def set_transform(self, transform: Transform) -> None:
        self.transform = transform

    def get_transform(self) -> Transform:
        return self.transform

    def get_inverse_transform(self, transform: Optional[Transform] = None) -> Transform:
        return Invertd(
            keys=["image", "label", "pred"],
            transform=transform or self.transform,
            orig_keys=["image", "label", "pred"],
            meta_keys=["image_meta_dict", "label_meta_dict", "pred_meta_dict"],
            orig_meta_keys=["image_meta_dict", "label_meta_dict", "pred_meta_dict"],
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        )

    def get_patient_id(self, idx: int) -> str:
        return self._reader.get_patient_id(idx)
    
    def patient_has_meta(self, idx: int) -> bool:
        return self._reader.patient_has_meta(idx)
    
    def get_patient_ids(self) -> Tuple[pd.Series, pd.Series]:
        return self._reader.get_patient_ids()


class MetaSubset(data.Subset, MetaDatasetAbstract):
    def __init__(self, dataset: MetaDataset, indices: Sequence[str], transform: Optional[Transform] = None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx: int) -> Union[MetaFinalItem, List[MetaFinalItem]]:
        return self.dataset.get_item(self.indices[idx], self.transform)

    def get_patient_id(self, idx: int) -> str:
        return self.dataset.get_patient_id(idx)

    def patient_has_meta(self, idx: int) -> bool:
        return self.dataset.patient_has_meta(idx)

    def set_transform(self, transform: Transform) -> None:
        self.transform = transform

    def get_transform(self) -> Transform:
        return self.transform

    def get_inverse_transform(self) -> Transform:
        return self.dataset.get_inverse_transform(self.transform)
    
    @classmethod
    def from_subset(cls, subset: data.Subset) -> "MetaSubset":
        return cls(subset.dataset, subset.indices)
    
    def get_patient_ids(self) -> Tuple[pd.Series, pd.Series]:
        series = self.dataset.get_patient_ids()

        indices = []
        for patient_id in self.indices:
            indices.append(series[0].index[series[0] == patient_id][0])

        patient_ids = series[0][indices]
        has_meta = series[1][indices]

        return (patient_ids, has_meta)
