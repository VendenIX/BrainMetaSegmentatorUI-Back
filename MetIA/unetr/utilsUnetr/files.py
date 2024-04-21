"""This module contains the needed functions, methods and classes
for mainulating files and folders.
"""

import glob
import os
from typing import Tuple

import numpy as np


def get_last_and_best_checkpoint_files(checkpoint_dir: str) -> Tuple[str, str]:
    """Retrieves the latest and best model checkpoint files.
    
    Arguments:
        checkpoint_dir: Path of the directory where model checkpoints are saved.
    
    Returns:
        last_checkpoint_file: Path of the latest checkpoint file.
        best_checkpoint_file: Path of the best checkpoint file.
    """
    best_checkpoint_file = None
    last_checkpoint_file = None
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint*.ckpt"))

    if len(checkpoint_files) != 0:
        # get last
        epochs = np.array([int(file.split("-")[1][6:]) for file in checkpoint_files])
        idx = np.argmax(epochs)
        last_checkpoint_file = checkpoint_files[idx]

        # get best
        losses = np.array([float(file.split("-")[-1][9:-5]) for file in checkpoint_files])
        idx = np.argmin(losses)
        best_checkpoint_file = checkpoint_files[idx]
    
    return last_checkpoint_file, best_checkpoint_file
