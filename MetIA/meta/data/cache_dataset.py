from copy import deepcopy
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Sequence, Union

from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform
from tqdm import tqdm

from .dataset import MetaDataset, MetaSubset
from .type_definition import MetaFinalItem, MetaIntermediateItem


class CacheMetaSubset(MetaSubset):
    """This class is a subclass of MetaSubset wich creates subset with patient who have Meta"""
    def __init__(self, dataset: MetaDataset, indices: Sequence[str],
                 transform: Transform, num_workers: Optional[int] = None) -> None:
        """initialises the class CacheMetaSubset and the data variable
        Args:
        dataset: the Meta dataset
        indices: a sequence of integer which representes the indices of data"""
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        
        super().__init__(dataset, indices, transform)

        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(int(self.num_workers), 1)
        self._cache: List = self._fill_cache()
    
    def __getitem__(self, idx: int) -> Union[MetaFinalItem, List[MetaFinalItem]]:
        """to get the data in the given index
        if this index is not on our data, we get the item in the class MetaSubset
        Args:
        idx: index of the wanted data
        Returns: the data on the given index"""
        data = self._transform(idx)
        return self.dataset.apply_end_transformation(data)
    
    @classmethod
    def from_meta_subset(cls, subset: MetaSubset, num_workers: Optional[int] = None) -> "CacheMetaSubset":
        """ function which creates an instances of the CacheMetaSubset class
        Args:
        subset: the data set of the class MetaSubset
        Returns: an instance of the current class"""
        return cls(subset.dataset, subset.indices, transform=subset.transform, num_workers=num_workers)

    def _fill_cache(self) -> List[MetaIntermediateItem]:
        with ThreadPool(self.num_workers) as p:
            return list(
                tqdm(
                    p.imap(self._load_cache_item, range(len(self.dataset))),
                    total=len(self.dataset),
                    desc="Loading dataset",
                    leave=False,
                )
            )

    def _load_cache_item(self, idx: int) -> MetaIntermediateItem:
        """
        Args:
            idx: the index of the input data sequence.
        """
        dict_object, patient_id, has_meta = self.dataset.get_item_without_transform(idx)
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.transform.transforms:  # type:ignore
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            dict_object = apply_transform(_xform, dict_object)
        return MetaIntermediateItem(dict_object, patient_id, has_meta)

    def _transform(self, index: int) -> MetaIntermediateItem:
        while index < 0: # support negative index
            index += len(self.dataset)

        # load data from cache and execute from the first random transform
        start_run = False
        if self._cache is None:
            self._cache = self._fill_cache()
        dict_object, patient_id, has_meta = self._cache[index]
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.transform.transforms:
            if start_run or isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                # only need to deep copy data on first non-deterministic transform
                if not start_run:
                    start_run = True
                    dict_object = deepcopy(dict_object)
                dict_object = apply_transform(_transform, dict_object)
        return MetaIntermediateItem(dict_object, patient_id, has_meta)
