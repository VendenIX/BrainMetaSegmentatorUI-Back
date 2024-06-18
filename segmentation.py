from typing import List, Optional
import pydicom
from io import BytesIO
import os
import nibabel as nib
import numpy as np
from unetr.utilsUnetr.transforms import CropBedd, RandCropByPosNegLabeld, ResizeOrDoNothingd
from monai.transforms import Compose, Orientationd, ScaleIntensityRanged, CropForegroundd, ToTensord
from monai.transforms import RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd
import torch
from monai.inferers import sliding_window_inference
from unetr.model_module import SegmentationTask
import monai.transforms as transforms
from rt_utils import RTStructBuilder
import scipy as sp
from torch.cuda.amp import autocast, GradScaler
from shapely.geometry import Polygon
from dicompylercore import dicomparser

COLORS = [
    [255, 0, 0],     # Rouge
    [0, 255, 0],     # Vert
    [0, 0, 255],     # Bleu
    [255, 255, 0],   # Jaune
    [0, 255, 255],   # Cyan
    [255, 0, 255],   # Magenta
    [192, 192, 192], # Gris
    [128, 0, 0],     # Bordeaux
    [128, 128, 0],   # Olive
    [0, 128, 0],     # Vert foncé
    [128, 0, 128],   # Pourpre
    [0, 128, 128],   # Sarcelle
    [0, 0, 128],     # Bleu marine
    [255, 165, 0],   # Orange
    [255, 20, 147],  # Rose vif
    [75, 0, 130],    # Indigo
    [255, 192, 203], # Rose pâle
    [70, 130, 180],  # Bleu acier
    [240, 230, 140], # Kaki
    [95, 158, 160]   # Vert cadet
]

"""
Transforme une liste de datasets DICOM en une image Nifti1Image
:param dicom_datasets: une liste de datasets DICOM
:return: une image Nifti1Image
"""
def dicom_to_nifti_in_memory(dicom_datasets: List[pydicom.Dataset]) -> nib.Nifti1Image:
    image_slices = [ds.pixel_array for ds in dicom_datasets]
    volume_3d = np.stack(image_slices, axis=-1)
    affine = np.eye(
        4)  # necessaire pour les algorithmes de traitement d'images médicales, ça permet de savoir comment les voxels sont disposés dans l'espace
    nifti_image = nib.Nifti1Image(volume_3d, affine)
    return nifti_image


"""
Applique le modèle de segmentation UNETR sur une image Nifti1Image
"""
def getLabelOfIRM_from_nifti(nifti_image: nib.Nifti1Image, pathModelFile: str):
    transform = transformation()
    transformed_image = applyTransforms(transform, nifti_image.get_fdata())

    model = loadModel(pathModelFile)
    # dico_image = {"image": transformed_image, "label": torch.zeros_like(transformed_image)}
    dico_image = applyUNETR(transformed_image, model)

    label, imageT = disapplyTransforms(transform, dico_image)

    labeled_array, num_features = sp.ndimage.label(label)

    print("Le modèle à trouvé ", num_features, " rois")
    # Trouver les slices pour chaque région
    slices = sp.ndimage.find_objects(labeled_array)
    # Parcourir chaque feature pour évaluer et potentiellement supprimer les petites régions
    for i in range(num_features):
        current_slice = slices[i]
        if current_slice is not None:
            # Extraire la région actuelle en utilisant la slice
            current_region = labeled_array[current_slice]

            # Calculer la taille de la région actuelle
            size = np.sum(current_region == (i + 1))
            print(f"Feature {i + 1}: Size = {size}")

            # Vérifier si la taille est inférieure au seuil
            if size < 100:
                # Définir les valeurs de cette région à zéro
                labeled_array[current_slice][current_region == (i + 1)] = 0
    label, num_features = sp.ndimage.label(labeled_array)
    print("Après deletion des petites rois, il reste", num_features, " rois")

    return nifti_image.get_fdata() / 255, label, imageT


"""
Application les transformations sur l'image
:param transform: les transformations à appliquer
:param image: les images à transformer
"""


def applyTransforms(transform, image):
    # Assurez-vous que l'image est un tensor PyTorch
    image = torch.tensor(image, dtype=torch.float32)

    image = (image / torch.max(image)) * 255
    image = image.unsqueeze(0)
    data = {"image": image, "label": torch.zeros_like(image)}
    transformed = transform(data)
    return transformed


"""
Transformations à appliquer sur les images avant de les passer au modèle de segmentation
"""


def transformation():
    dtype = torch.float32
    voxel_space = (1.5, 1.5, 2.0)
    a_min = -200.0
    a_max = 300
    b_min = 0.0
    b_max = 1.0
    clip = True
    crop_bed_max_number_of_rows_to_remove = 0
    crop_bed_max_number_of_cols_to_remove = 0
    crop_bed_min_spatial_size = (300, -1, -1)
    enable_fgbg2indices_feature = False
    pos = 1.0
    neg = 1.0
    num_samples = 1
    roi_size = (96, 96, 96)
    random_flip_prob = 0.2
    random_90_deg_rotation_prob = 0.2
    random_intensity_scale_prob = 0.1
    random_intensity_shift_prob = 0.1
    val_resize = (-1, -1, 250)

    spacing = transforms.Identity()
    if all([space > 0.0 for space in voxel_space]):
        spacing = transforms.Spacingd(
            keys=["image", "label"], pixdim=voxel_space, mode=("bilinear", "nearest")
        )  # to change the dimension of the voxel to have less data to compute

        posneg_label_croper_kwargs = {
            "keys": ["image", "label"],
            "label_key": "label",
            "spatial_size": roi_size,
            "pos": pos,
            "neg": neg,
            "num_samples": num_samples,
            "image_key": "image",
            "allow_smaller": True,
        }

        fgbg2indices = transforms.Identity()
        if enable_fgbg2indices_feature:
            fgbg2indices = transforms.FgBgToIndicesd(
                keys=["image", "label"], image_key="label", image_threshold=0.0
            )  # to crop samples close to the label mask
            posneg_label_croper_kwargs["fg_indices_key"] = "image_fg_indices"
            posneg_label_croper_kwargs["bg_indices_key"] = "image_bg_indices"
        else:
            posneg_label_croper_kwargs["image_threshold"] = 0.0

    transform = transforms.Compose(
        [
            transforms.Orientationd(keys=["image", "label"], axcodes="LAS", allow_missing_keys=True),
            # to have the same orientation
            spacing,
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip, allow_missing_keys=True
            ),  # scales image from a values to b values
            CropBedd(
                keys=["image", "label"], image_key="image",
                max_number_of_rows_to_remove=crop_bed_max_number_of_rows_to_remove,
                max_number_of_cols_to_remove=crop_bed_max_number_of_cols_to_remove,
                min_spatial_size=crop_bed_min_spatial_size,
                axcodes_orientation="LAS",
            ),  # crop the bed from the image (useless data)
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True),
            # remove useless background image part
            fgbg2indices,
            transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=0,
                                 allow_missing_keys=True),  # random flip on the X axis
            transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=1,
                                 allow_missing_keys=True),  # random flip on the Y axis
            transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=2,
                                 allow_missing_keys=True),  # random flip on the Z axis
            transforms.RandRotate90d(keys=["image", "label"], prob=random_90_deg_rotation_prob, max_k=3,
                                     allow_missing_keys=True),  # random 90 degree rotation
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=random_intensity_scale_prob),
            # random intensity scale
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=random_intensity_shift_prob),
            # random intensity shifting
            transforms.ToTensord(keys=["image", "label"], dtype=dtype),  # to have a PyTorch tensor as output
        ]
    )
    return transform


"""
Applique le modèle de segmentation UNETR sur les images dicoms converties en nifti
:param dicoImage: un dictionnaire contenant les images et les labels
:param model: le modèle de segmentation UNETR
"""


def applyUNETR(dicoImage, model):
    scaler = GradScaler()
    with torch.no_grad():
        with autocast():
            label = sliding_window_inference(inputs=dicoImage["image"][None],
                                            roi_size=(96, 96, 96),
                                            sw_batch_size=1,
                                            predictor=model,
                                            overlap=0) #0.47

    label = torch.argmax(label, dim=1, keepdim=True)

    size = label.shape
    print("applyUNETR", size[1], size[2], size[3], size[4])
    dicoImage["label"] = label.reshape((size[1], size[2], size[3], size[4]))
    return dicoImage


"""
Enlève les transformations appliquées à l'image
"""


def disapplyTransforms(transform, dicoImage):
    dicoImage = transform.inverse(dicoImage)
    return dicoImage["label"], dicoImage["image"]


"""
Chargement du modèle pré-entrainé pour la segmentation
"""


def loadModel(pathModelFile):
    # map_location = torch.device('cpu') a faire pas ici mais dans unetr/model_module.py ligne 133 ou try faire ailleurs
    model = SegmentationTask.load_from_checkpoint(pathModelFile)#.to('cuda')
    model.eval()
    return model


"""
Permet d'écrire sur un RTStruct via les dicoms, un label
:param dicom_datasets: les images dicoms concernées par le RTStruct
:param label: le label obtenu via le modèle
"""
def process_rtstruct_and_calculate_details(dicom_datasets, label, existing_rtstruct=None, voxel_dimensions=(0.5, 0.5, 1.0)):
    print("shape 1")
    print(label.shape)
    if existing_rtstruct:
        rtstruct = RTStructBuilder.create_from_memory(dicom_datasets, existing_rtstruct)
        isFromCurrentRTStruct = True
    else:
        rtstruct = RTStructBuilder.create_new_from_memory(dicom_datasets)
        isFromCurrentRTStruct = False

    results = []
    labeled_array, num_features = sp.ndimage.label(label)
    print("shape 2")
    objects = sp.ndimage.find_objects(labeled_array)

    for i in range(1, num_features + 1):
        mask = np.where(label[0, :, :, :] == i, True, False) 
        region_volume = np.sum(mask) * np.prod(voxel_dimensions)  # Volume in mm³
        bbox_lengths = [extent.stop - extent.start for extent in objects[i-1]]
        diameters = [length * voxel for length, voxel in zip(bbox_lengths, voxel_dimensions)]
        start_slice, end_slice = objects[i-1][2].start, objects[i-1][2].stop  # Z-dimension slices

        # Add ROI to RTStruct
        color = COLORS[(i - 1) % len(COLORS)]
        roi_name = f"GTV_MetIA_{i}"
        if existing_rtstruct:
            roi_name = generate_unique_name(rtstruct, f"GTV_MetIA_{i}")
        rtstruct.add_roi(mask=mask, color=color, name=roi_name)

        # Save details
        results.append({
            'Region ID': i,
            'Volume (mm³)': region_volume,
            'Diameters (mm)': diameters,
            'Start Slice': start_slice,
            'End Slice': end_slice
        })

    return rtstruct, results, isFromCurrentRTStruct


def generate_unique_name(rtstruct, base_name):
    """
    Generate a unique name for an ROI by appending a suffix if the name already exists in the RTStruct.

    Args:
        rtstruct (RTStruct): The RTStruct object where the ROI will be added.
        base_name (str): The base name for the ROI.

    Returns:
        str: A unique name for the ROI.
    """

    """
    existing_names = rtstruct.get_roi_names()
    suffix = 1
    unique_name = base_name
    while unique_name in existing_names:
        unique_name = f"{base_name}_{suffix}"
        suffix += 1
    return unique_name
    """
    existing_names =  rtstruct.get_roi_names()  # Get all existing ROI names
    new_names = set()  # Pour garder les noms générés dans cette session
    suffix = 1

    # Vérifier d'abord si un nom de base conflictuel existe
    conflict = any(name.startswith(base_name) for name in existing_names)

    # Si un conflit existe, ajuster tous les noms
    if conflict:
        for name in existing_names:
            if name.startswith(base_name):
                new_name = f"{base_name}_{suffix}"
                while new_name in existing_names or new_name in new_names:
                    suffix += 1
                    new_name = f"{base_name}_{suffix}"
                new_names.add(new_name)
    else:
        # Aucun conflit, utiliser le nom de base directement
        new_names.add(base_name)

    # Renvoyer le dernier nom unique généré ou le nom de base si aucun conflit n'était présent
    return max(new_names, key=len)

def generate_rtstruct_segmentation_unetr(dicom_datasets: List[pydicom.dataset.Dataset], pathModelFile: str,
                                         existing_rtstruct: Optional[pydicom.dataset.Dataset] = None):
    """
    Appel le modèle pour générer un RTStruct

    Args :
        dicom_datasets: les images dicoms
        pathModelFile : path du modele
        existing_rtstruct : le rtstruct sur lequel on se base (optionnel, on peut ne pas en mettre)

    Returns:
        Dataset, Boolean: Le RTStruct correspondant à la segmentation, Est ce que c'est un RTStruct update ou create (faut il remplacer un précédant RTStruct par celui-ci)
    """
    niftis = dicom_to_nifti_in_memory(dicom_datasets)
    image, label, imageT = getLabelOfIRM_from_nifti(niftis, pathModelFile)
    rt_struct, metastases_details, isFromCurrentRTStruct = process_rtstruct_and_calculate_details(dicom_datasets, label, existing_rtstruct)
    print("Tout s'est bien passé on dirait")
    for detail in metastases_details:
        print(detail)
    return rt_struct, isFromCurrentRTStruct

def extract_roi_info(rtstruct, dicom_series):

    dicom_series.sort(key=lambda x: int(x.InstanceNumber))
    
    # Extraire les informations de PixelSpacing et SliceThickness
    pixel_spacing = dicom_series[0].PixelSpacing
    slice_thickness = dicom_series[0].SliceThickness

    # Créer une map des positions de slices à leurs indices
    slice_positions = {round(dcm.ImagePositionPatient[2], 2): i+1 for i, dcm in enumerate(dicom_series)}

    rtstruct_infos = {
        "PatientName": rtstruct.PatientName,
        "PatientID": rtstruct.PatientID,
        "PatientBirthDate": rtstruct.PatientBirthDate,
        "PatientSex": rtstruct.PatientSex,
        "StudyDate" : rtstruct.StudyDate,
        "StudyInstanceUID" : rtstruct.StudyInstanceUID
    }
    # Charger le fichier RTStruct avec dicompyler-core
    rtstruct = dicomparser.DicomParser(rtstruct)

    # Extraire les structures
    structures = rtstruct.GetStructures()

    # Fonction pour calculer le diamètre d'un ROI
    def calculate_diameter(contour):
        max_distance = 0
        for i in range(len(contour)):
            for j in range(i + 1, len(contour)):
                distance = np.linalg.norm(np.array(contour[i]) - np.array(contour[j]))
                if distance > max_distance:
                    max_distance = distance
        return max_distance

    # Fonction pour calculer le volume d'un ROI
    def calculate_volume(coords, thickness):
        volume = 0
        for z in coords.keys():
            contours = coords[z]
            for contour in contours:
                if len(contour['data']) >= 4:
                    polygon = Polygon(contour['data'])
                    volume += polygon.area * thickness
        return volume / 1000  # Convertir en cm³

    # Initialiser le dictionnaire de résultats
    roi_info = {}

    # Parcourir les structures et extraire les informations
    for roi_number, roi_data in structures.items():
        roi_name = roi_data['name']
        
        # Obtenir les coordonnées des contours
        coords = rtstruct.GetStructureCoordinates(roi_number)
        
        # Calculer le volume
        thickness = dicomparser.DicomParser.CalculatePlaneThickness(rtstruct, coords)
        volume = calculate_volume(coords, thickness)
        
        # Calculer le diamètre maximal
        contours = []
        contour_slice_indices = []
        for plane in coords.values():
            for contour in plane:
                contours.append(contour['data'])
                z_pos = round(contour['data'][0][2], 2)
                if z_pos in slice_positions:
                    contour_slice_indices.append(slice_positions[z_pos])
        
        diameters = [calculate_diameter(contour) for contour in contours]
        nb_dicoms = len(dicom_series)
        
        # Obtenir les slices de début et de fin
        start_slice = nb_dicoms - max(contour_slice_indices) if contour_slice_indices else None
        end_slice = nb_dicoms - min(contour_slice_indices) if contour_slice_indices else None

        # Ajouter les informations au dictionnaire
        roi_info[roi_name] = {
            "diameter_max": max(diameters),
            "volume_cm3": volume,
            "start_slice": start_slice,
            "end_slice": end_slice
        }


    
    return roi_info, rtstruct_infos

if __name__ == "__main__":
    ############################################################################################################
    # MOCK API
    def load_dicom_files_from_directory(directory_path):
        dicom_datasets = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".dcm"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'rb') as file:
                    dicom_file = pydicom.dcmread(BytesIO(file.read()))
                    dicom_datasets.append(dicom_file)
        return dicom_datasets


    pathSlicesIRM = '/Users/romain/Documents/P_R_O_J_E_C_T_S/IRM-Project/mbiaDataDownloads/DATA_VERITE_TERRAIN/RM'
    pathModelFile = '/Users/romain/Downloads/Modeles_Pre_Entraines/checkpoint_epoch1599_val_loss0255.cpkt'
    dicom_datasets = load_dicom_files_from_directory(pathSlicesIRM)
    niftis = dicom_to_nifti_in_memory(dicom_datasets)

    image, label, imageT = getLabelOfIRM_from_nifti(niftis, pathModelFile)

    rt_struct = create_rtstruct(dicom_datasets, label)

    # Afficher les résultats
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Transformed image shape:", imageT.shape)