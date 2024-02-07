from typing import Any

import numpy as np
import torch

def np2torch(
    array: np.ndarray,
    device: torch.device = None
) -> torch.Tensor:
	"""Transform a numpy array to pytorch tensor
	Args:
		array: a numpy table that we want to transform to a torch tensor.
		device: specify the device where we want to create the tensor. If it's None, one the CPU, else we can create it on the GPU,CPU or TPU,... (it's an argument for the futur tensor)

	Returns:
		a tensor from a copy of "array" create on "device".
	"""
	return torch.tensor(array.copy(), device=device)


def torch2np(array: torch.Tensor) -> np.ndarray:
	"""Transform a pytorch tensor to a table of array

	Args:
		array: the torch.Tensor that we want to transform

	Returns:
		a table of array made with the "array"
	"""
	return array.numpy()
    

def all2np(array: Any) -> np.ndarray:
	""" transform something to a Numpy array
	At first, If "array" is an instance of a Numpy table and return "array"
	second, If "array" is a pytorch tensor, transform it to a numpy array
	finaly, transform "array" which is an other type of data to a numpy array

	Args:
		array: a data with a unknown type that we want to transform to a numpy array
	Returns:
		"array" transform to a numpy array
	"""
	if isinstance(array, np.ndarray):
		return array    
	if torch.is_tensor(array):
		return torch2np(array)

	return np.array(array)

def all2torch(array: Any) -> torch.Tensor:
	""" transform something to a torch tensor
	At first, If "array" is a torch tensor, return "array"
	finaly, transform "array" which is an other type of data to a torch tensor

	Args:
		array: a data with a unknown type that we want to transform to a torch tensor
	Returns:
		"array" transform to a torch tensor
	"""
	if torch.is_tensor(array):
		return array
	return torch2np(array)
