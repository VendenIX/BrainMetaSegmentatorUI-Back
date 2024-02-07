import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
	"""set the seed to a value, permite to have the same value for python, numpy, pytorch and cuda.

	Args:
		seed: the value for the seed that we want to use
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.random.seed(seed)
	torch.cuda.random.seed(seed)


def create_generator(seed: int) -> torch.Generator:
	"""create a generater of aleatoirs number, the sequence of aleatoires number will be determinate by "seed"
	Args:
		seed: the value for the seed that we want to use
	Returns:
		an aleatoirs generator number
	"""
	return torch.Generator().manual_seed(seed)
