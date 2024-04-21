import os
import time as t

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
from pydicom.encaps import encapsulate
from MetIA.Modele import Net 

import nibabel as nib

PATH_MODEL = './MetIA/resultsult/'

def simulate_modele_transform(file_paths):
    modified_dicoms = []  # Liste pour stocker les objets DICOM modifiés

    for file_path in file_paths:
        # Chargement du fichier DICOM
        dicom_data = pydicom.dcmread(file_path)

        # Extraction de l'image des données DICOM
        image = dicom_data.pixel_array
        
        # Application du seuillage simple
        thresholded_image = np.where(image > 1000, image, 0)

        # Vérifiez si les données sont compressées et encapsulez-les si nécessaire
        if dicom_data.file_meta.TransferSyntaxUID in [pydicom.uid.JPEGBaseline, pydicom.uid.JPEGExtended, pydicom.uid.JPEG2000]:
            # Encapsulate the modified data
            dicom_data.PixelData = encapsulate([thresholded_image.tobytes()])
        else:
            dicom_data.PixelData = thresholded_image.tobytes()

        dicom_data.Rows, dicom_data.Columns = thresholded_image.shape
        
        # Ajout de l'objet DICOM modifié à la liste
        modified_dicoms.append(dicom_data)
    
    # Retourne la liste des données DICOM modifiées
    return modified_dicoms


def simulate_rtstruct_generation():
    print()
    # Attendre 30 secondes pour simuler le traitement des images DICOM
    print("Simulating RTStruct generation...")
    t.sleep(5)


    # Simulation d'un traitement des images DICOM et génération d'un RTStruct
    # Mettez le chemin absolue de votre RTStruct ici pour simuler l'envoi
    rtstruct_path = '/home/romain/Documents/P_R_O_J_E_C_T_S/projetIRM/BrainMetaSegmentatorUI-Back/1-1.dcm'
    return rtstruct_path


def prediction(ArrayDicoms):
    # conversion des dicoms en nifti 
    # ArrayDicoms : liste des dicoms
    # prediction : liste des nifti
    prediction = []
    for dicom in ArrayDicoms:
        # co

    # prediction du modele, recharger le modele : 
    model = Net()
    model.load_state_dict(torch.load(PATH_MODEL)) 
    model.eval()
    # prediction : application du modele sur les images converties en nifti

    return prediction
