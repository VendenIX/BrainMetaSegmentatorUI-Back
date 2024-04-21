import warnings

import torch

from .jupyter_notebooks import is_notebook


def get_device(use_gpu: bool = True, use_tpu: bool = False):
	"""Return the device that we want to use and which is available on the used computer
	
	create a variable name "device_name" with the value "cpu"
	if the user say True for the variable "use_tpu", import packages that provides support for running PyTorch on TPU and return the associate device
	after if the user say True for "use_gpu", verify that a gpu is available and, if it's not, return a warning message.
																				  if it's, change the value of "device_name" to "cuda"
	finaly, return "device_name" transform to a torch device.
	
	Args:
		use_gpu: a boolean which say if we want to use a gpu or not
		use_tpu: a boolean which say if we want to use a tpu or not
	Returns:
		the device compatible with the used computer

	Raises:
			
	See also:
			
	Notes:

	References:
			
	Examples:
	"""
	if use_tpu:
		import torch_xla
		import torch_xla.core.xla_model as xm
		return xm.xla_device()

	device_name = "cpu"
	if use_gpu and not torch.cuda.is_available():
		warnings.warn("you can't use GPU because you doesn't have a GPU or check your version of PyTorch")
	elif use_gpu:
		device_name = "cuda"

	return torch.device(device_name)


#check if the code use here is a notebook.
#if it's import nest_asyncio which is use for helping to run asynchronous code on a jupyter notebook and apply it.
if is_notebook():
    import nest_asyncio
    nest_asyncio.apply()
