from typing import Sequence, Union

import numpy as np
import torch


class TensorUtils:
    """Class that contains some utility methods
    for PyTorch tensors."""

    @staticmethod
    def clear_gpu_memory() -> None:
        """Clears the GPU memory."""
        torch.cuda.empty_cache()

    @classmethod
    def move_tensors_to_cpu(cls, *tensors: torch.Tensor, to_numpy: bool = False) -> Sequence[Union[torch.Tensor, np.ndarray]]:
        """Moves the given tensors to the CPU.
        
        Arguments:
            *tensors: Tensors to move.
            to_numpy: Boolean that enable the conversion of the tensors to NumPy N-Dimensional arrays.
        
        Returns:
            output_tensors: Tensors moved in the CPU.
        
        Note:
            It also support NoneType tensors to easily give a list of tensors.
        """
        outputs = [None]*len(tensors)

        # move the tensors to the CPU
        for ii, tensor in enumerate(tensors):
            if tensor is not None:
                outputs[ii] = tensor.detach().cpu()
        
        # remove cache from the GPU
        cls.clear_gpu_memory()

        if to_numpy:
            outputs = cls.convert_to_numpy_array(outputs)
        
        return outputs
    
    @classmethod
    def convert_to_numpy_array(cls, *tensors: torch.Tensor) -> Sequence[np.ndarray]:
        """Convert the tensors to NumPy N-Dimensional arrays.
        
        Arguments:
            *tensors: Tensors to convert.
        
        Returns:
            output_arrays: Converted tensors.
        
        Note:
            It also support NoneType tensors to easily give a list of tensors.
        """
        outputs = [None]*len(tensors)

        for ii, tensor in enumerate(tensors):
            if tensor is not None:
                outputs[ii] = tensor.detach().cpu().numpy()

        return outputs
    
    @staticmethod
    def convert_to(*arrays_or_tensors: Union[torch.Tensor, np.ndarray], dtype) -> Sequence[Union[torch.Tensor, np.ndarray]]:
        """Converts NumPy arrays or PyTorch tensors to the same type as input
        but with a different dtype as output.
        
        Arguments:
            *arrays_or_tensors: Arrays or tensors to convert.
        
        Returns:
            outputs: Converted arrays/tensors.
        
        Note:
            It also support NoneType arrays/tensors to easily give a list of arrays/tensors.
        """
        outputs = [None]*len(arrays_or_tensors)

        for ii, array_or_tensor in enumerate(arrays_or_tensors):
            if array_or_tensor is not None:
                if isinstance(array_or_tensor, torch.Tensor):
                    outputs[ii] = array_or_tensor.to(dtype)
                else:
                    outputs[ii] = array_or_tensor.astype(dtype)

        return outputs
