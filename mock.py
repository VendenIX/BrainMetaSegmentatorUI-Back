import os
import sys
import time as t
from typing import List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
import torch
from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate
from scipy.ndimage import gaussian_filter, sobel

from rt_utils import RTStructBuilder

PATH_MODEL = './MetIA/resultsult/'

# pour détecter les contours d'une image DICOM
def detect_contours(image):
    """
    Si tu ne comprends rien :  
    Un filtre de Sobel est appliqué pour trouver les gradients dans les deux directions x et y. 
    Le gradient total (magnitude) est calculé et un seuil est appliqué pour obtenir un masque de contour.
    """
    
    # lissage de l'image pour réduire le bruit avant la détection des contours
    smoothed_image = gaussian_filter(image, sigma=2)
    
    # calcul du gradient de l'image lissée
    sx = sobel(smoothed_image, axis=0, mode='constant')
    sy = sobel(smoothed_image, axis=1, mode='constant')
    sobel_mag = np.hypot(sx, sy)
    
    # seuil de détection du contour
    threshold = np.percentile(sobel_mag, 95)
    
    # matrice boolenne
    return sobel_mag > threshold

"""
Simulation de la génération d'un RTStruct à partir de fichiers DICOM passés en mémoire
"""
def simulate_rtstruct_generation2(dicom_datasets: List[Dataset], existing_rtstruct: Dataset):

    dicom_files = dicom_datasets
    # lire et trier les fichiers dicoms par 'InstanceNumber' #si c pas fait, les rois corespondront pas aux dicoms
    dicom_files.sort(key=lambda x: x.InstanceNumber if 'InstanceNumber' in dir(x) else 0)

    label = create_mask_from_dicom(dicom_datasets)
    if existing_rtstruct is not None:
        rtstruct, isFromCurrentRTStruct = update_rtstruct(dicom_datasets, existing_rtstruct, label) 
        isFromCurrentRTStruct = True
    else:
        rtstruct, isFromCurrentRTStruct = create_rtstruct(dicom_datasets, label)

    return rtstruct, isFromCurrentRTStruct

def create_rtstruct(dicom_datasets: List[Dataset], label):
    """
    Crée un nouveau RTStruct à partir des fichiers DICOM et des masques générés.
    """
    print("Je crérer un nouveau RTStruct pour le mock")
    rtstruct = RTStructBuilder.create_new_from_memory(dicom_datasets)
    mask = create_mask_from_dicom(dicom_datasets)
    rtstruct.add_roi(mask=mask, color=[255, 0, 0], name='Brain Contours')
    return rtstruct, False

def update_rtstruct(dicom_datasets: List[Dataset], existing_rtstruct: Dataset, label):
    """
    Met à jour un RTStruct existant avec de nouveaux contours basés sur les fichiers DICOM fournis.
    """
    print("Je réécris sur le RTStruct déjà présent pour le mock")
    rtstruct = RTStructBuilder.create_from_memory(dicom_datasets, existing_rtstruct)
    mask = create_mask_from_dicom(dicom_datasets)
    rtstruct.add_roi(mask=mask, color=[255, 0, 0], name='Updated Brain Contours')
    return rtstruct, True

def create_mask_from_dicom(dicom_datasets: List[Dataset]):
    """
    Crée un masque 3D à partir des fichiers DICOM en utilisant la fonction de détection de contours.
    """
    dicom_datasets.sort(key=lambda x: x.InstanceNumber if 'InstanceNumber' in dir(x) else 0)
    num_slices = len(dicom_datasets)
    mask_shape = (dicom_datasets[0].Rows, dicom_datasets[0].Columns, num_slices)
    full_mask = np.zeros(mask_shape, dtype=bool)

    for i, dicom_data in enumerate(dicom_datasets):
        image = dicom_data.pixel_array
        contour_mask = detect_contours(image)
        full_mask[:, :, i] = contour_mask

    return full_mask