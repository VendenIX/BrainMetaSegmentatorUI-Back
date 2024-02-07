from unetr.model_module import SegmentationTask
import pydicom
import os
import numpy as np




"""
    Classe permettant de charger un modèle de segmentation à partir du fichier spécifié.

    Args:
        pathModelFile (str): Chemin vers le fichier contenant le modèle.

    Returns:
        SegmentationTask: Modèle de segmentation chargé et prêt pour l'inférence.
"""

class ModelLoader:
    def __init__(self, path_model_file):
        self.path_model_file = path_model_file

    def load_model(self):
        model = SegmentationTask.load_from_checkpoint(self.path_model_file)
        model.eval()
        return model
    

"""
    Charge une série d'images DICOM à partir du dossier spécifié.

    Args:
        slices_folder (str): Chemin vers le dossier contenant les fichiers DICOM.

    Returns:
        np.ndarray: Tableau contenant les images DICOM chargées.
"""

class DicomImageLoader:
    def __init__(self, slices_folder):
        self.slices_folder = slices_folder

    def load_dicom_image(self):
        slices = [pydicom.dcmread(os.path.join(self.slices_folder, f)) for f in os.listdir(self.slices_folder)]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        image = np.stack([s.pixel_array for s in slices])
        return image
