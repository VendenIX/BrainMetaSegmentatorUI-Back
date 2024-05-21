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
def simulate_rtstruct_generation2(dicom_datasets: List[Dataset]):

    dicom_files = dicom_datasets
    # lire et trier les fichiers dicoms par 'InstanceNumber' #si c pas fait, les rois corespondront pas aux dicoms
    dicom_files.sort(key=lambda x: x.InstanceNumber if 'InstanceNumber' in dir(x) else 0)
    # creation du rtstruct vierge
    rtstruct = RTStructBuilder.create_new_from_memory(dicom_datasets)


    # init du masque 3D basé sur le nombre de fichiers et la taille des images
    num_slices = len(dicom_files)
    mask_shape = (dicom_files[0].Rows, dicom_files[0].Columns, num_slices)
    full_mask = np.zeros(mask_shape, dtype=bool)

    # pour chaque fichier dicom :
    for i, dicom_data in enumerate(dicom_files):
        # calculer le masque correspondant et le mettre dans le masque 3D
        image = dicom_data.pixel_array
        contour_mask = detect_contours(image)
        full_mask[:, :, i] = contour_mask  # remplissage du masque 3D

    # ajouter le masque au fichier RTStruct
    rtstruct.add_roi(mask=full_mask, color=[255, 0, 0], name='Brain Contours')

    # sauvegarde du fichier rtstruct
    return rtstruct