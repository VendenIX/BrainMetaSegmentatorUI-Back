import os
import shutil
import tempfile
from io import BytesIO
from typing import List

import pydicom
import requests
from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS
from pydicom.dataset import Dataset

from BDD.MesuresSQLite import MesuresDB
from mock import simulate_rtstruct_generation2


from MetIA.Model2 import Net

import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImage, ToTensor

import torch

load_dotenv()


# def load_adjusted_checkpoint(filepath, model):
#     # Charger le checkpoint
#     checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

#     # Extraire le state_dict du checkpoint
#     state_dict = checkpoint['state_dict']

#     # Ajuster le state_dict si nécessaire
#     new_state_dict = {}
#     for key, value in model.state_dict().items():
#         # Supprimer un préfixe spécifique (dans ce cas, 'backbone.0.') des clés
#         adjusted_key = 'backbone.0.' + key
#         if adjusted_key in state_dict:
#             new_state_dict[key] = state_dict[adjusted_key]
#         else:
#             # Utiliser la valeur originale si la clé ajustée n'existe pas
#             new_state_dict[key] = value

#     # Charger le state_dict ajusté dans le modèle
#     model.load_state_dict(new_state_dict, strict=False)
#     model.eval()
#     return model


app = Flask(__name__)
CORS(app)
#ORTHANC_URL = os.getenv("ORTHANC_URL")

ORTHANC_URL = "http://localhost:8042/"

# # Modele
# model = Net() 
# checkpoint_path = 'checkpoint-epoch=1599-val_loss=0.225.ckpt'
# net = load_adjusted_checkpoint(checkpoint_path, model)

@app.route('/getAllStudies', methods=['GET'])
def get_all_studies():
    try:
        response = requests.get(f"{ORTHANC_URL}/studies")
        if response.status_code == 200:
            studies = response.json()
            return jsonify(studies), 200
        else:
            return jsonify({"error": "Failed to retrieve studies from PACS server"}), response.status_code
    except requests.exceptions.RequestException as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/getStudy/<string:study_id>', methods=['GET'])
def get_study(study_id):
    try:
        response = requests.get(f"{ORTHANC_URL}/studies/{study_id}")
        if response.status_code == 200:
            study_details = response.json()
            return jsonify(study_details), 200
        else:
            return jsonify({"error": "Failed to retrieve study from PACS server"}), response.status_code
    except requests.exceptions.RequestException as e:
        print(e)
        return jsonify({"error": str(e)}), 500
    
@app.route('/getStudyDicoms/<study_id>', methods=['GET'])
def get_study_dicoms(study_id):
    try:
        # Récupération des instances DICOM pour l'étude spécifiée
        response = requests.get(f"{ORTHANC_URL}/studies/{study_id}/instances")
        if response.status_code == 200:
            dicom_instances = response.json()
            print("DICOM Instances for Study ID", study_id, ":", dicom_instances)
            return jsonify(dicom_instances), 200
        else:
            return jsonify({"error": "Failed to retrieve DICOM instances"}), response.status_code
    except requests.exceptions.RequestException as e:
        print(e)
        return jsonify({"error": "Server error"}), 500

@app.route('/uploadDicom', methods=['POST'])
def upload_dicom():
    print("Requête reçue pour /uploadDicom")
    file = request.files['file']
    print(f"Fichier reçu : {file.filename}")
    try:
        # Vérifier l'extension du fichier
        # C'est une approche basique, préférable d'analyser le contenu du fichier
        if not file.filename.endswith('.dcm'):
            return jsonify({"error": "Le fichier n'est pas un fichier DICOM (.dcm)"}), 400
        
        # Pour une validation plus robuste, tenter de lire le fichier comme un fichier DICOM
        print(f"Tentative de lecture du fichier DICOM : {file.filename}")
        ds = pydicom.dcmread(file)
        if ds is None or len(ds) == 0:
            print("Le fichier DICOM est vide ou non lisible.")
            return jsonify({"error": "Le fichier DICOM est vide ou non lisible."}), 400
        file.seek(0) # # Remettre le pointeur au début du fichier, c'est ultra important, sinon le pycom read empêche l'upload, sinon le fichier est à la fin et donc "vide" du point de vue de la lecture.

        print(f"Lecture du fichier DICOM réussie : {file.filename}")
        # Effectuer l'upload vers Orthanc ou autre logique nécessaire
        print(f"Préparation de l'envoi du fichier à Orthanc : {file.filename}")
        files = {'file': (file.filename, file, 'application/dicom')}
        response = requests.post(f"{ORTHANC_URL}/instances", files=files)
        print(f"Réponse d'Orthanc : Statut {response.status_code}, Contenu {response.content}")

        if response.status_code in [200, 202]:
            print("DICOM file uploaded successfully")
            return jsonify({"success": "DICOM file uploaded successfully"}), response.status_code
        else:
            print("Failed to upload DICOM file to Orthanc")
            return jsonify({"error": "Failed to upload DICOM file to Orthanc"}), response.status_code
    except Exception as e:
        print(f"Erreur lors du traitement du fichier DICOM : {e}")
        return jsonify({"error": "Erreur serveur"}), 500
    
@app.route('/delete-dicom-instance/<study_instance_uid>', methods=['DELETE'])
def delete_dicom_instance(study_instance_uid):
    orthanc_study_id = find_orthanc_study_id_by_study_instance_uid(study_instance_uid)
    print("Je supprime l'instance DICOM avec StudyInstanceUID", study_instance_uid, "et ID Orthanc", orthanc_study_id)

    try:
        # Envoie une requête DELETE à Orthanc
        response = requests.delete(f"{ORTHANC_URL}/instances/{orthanc_study_id}")
        print("---------------------------------")
        print(response)
        print("---------------------------------")
        # Vérifie si la suppression a réussi
        if response.status_code == 200:
            print("enfaite")
            return jsonify({"success": "Instance DICOM supprimée avec succès"}), 200
        else:
            print("bas non")
            print(response.text)
            print(response.status_code)
            return jsonify({
                "error": "Failed to delete DICOM instance",
                "status_code": response.status_code,
                "response_body": response.text
            }), response.status_code

    except requests.exceptions.RequestException as e:
        # Gestion des erreurs de connexion ou autres erreurs de réseau
        return jsonify({"error": "Erreur lors de la connexion à Orthanc", "exception": str(e)}), 500


@app.route('/segmentation/<study_instance_uid>', methods=['POST'])
def segmentation(study_instance_uid):
    # Convertir StudyInstanceUID en ID Orthanc
    orthanc_study_id = find_orthanc_study_id_by_study_instance_uid(study_instance_uid)
    
    if not orthanc_study_id:
        return jsonify({"error": "StudyInstanceUID not found"}), 404

    try:
        # Récupérer les métadonnées des instances DICOM pour l'étude spécifiée
        response = requests.get(f"{ORTHANC_URL}/studies/{orthanc_study_id}/instances")
        if response.status_code != 200:
            return jsonify({"error": "Failed to retrieve DICOM instances"}), response.status_code

        # Extraire les identifiants des instances DICOM à partir de la réponse JSON
        instances = response.json()
        dicom_data = []

        # Télécharger chaque fichier DICOM en utilisant son identifiant
        for instance in instances:
            instance_id = instance['ID']
            dicom_response = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/file", stream=True)

            if dicom_response.status_code == 200:
                # Vous pouvez ici charger les données DICOM directement dans pydicom si nécessaire
                dicom_file = pydicom.dcmread(BytesIO(dicom_response.content))
                dicom_data.append(dicom_file)
            else:
                print(f"Failed to download DICOM file for instance ID {instance_id}")

        print("DICOM files retrieved and processed successfully.")
        #print(dicom_data[0])
        rtstruct = simulate_rtstruct_generation2(dicom_data)  # mock
        print("mon rtstruct c'est cela")
        print(rtstruct)
        upload_rtstruct(rtstruct)

        # mise en place du modèle


        return jsonify({"success": "DICOM files retrieved and processed successfully."}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500





# @app.route('/segmentation/<study_instance_uid>', methods=['POST'])
# def segmentation(study_instance_uid):
#     orthanc_study_id = find_orthanc_study_id_by_study_instance_uid(study_instance_uid)
    
#     if not orthanc_study_id:
#         return jsonify({"error": "StudyInstanceUID not found"}), 404

#     try:
#         # Récupérer les instances DICOM
#         response = requests.get(f"{ORTHANC_URL}/studies/{orthanc_study_id}/instances")
#         if response.status_code != 200:
#             return jsonify({"error": "Failed to retrieve DICOM instances"}), response.status_code

#         # Convertir les données DICOM en format NIfTI, puis en tenseurs pour le modèle
#         instances = response.json()
#         input_tensor = prepare_model_input(instances)

#         print("reussi")

#         # Exécuter le modèle
#         with torch.no_grad():
#             model_output = net(input_tensor)

#         # Post-traiter les résultats pour la sortie
#         output_data = process_model_output(model_output)

#         return jsonify({"success": "Segmentation completed", "data": output_data}), 200
#     except requests.exceptions.RequestException as e:
#         return jsonify({"error": f"Server error: {str(e)}"}), 500


# def convert_dicom_to_nifti(dicom_instances):
#     # Ce code suppose que vous avez déjà un moyen de convertir les DICOM en NIfTI
#     # Exemple simplifié de la conversion
#     dicom_data = []
#     for instance in dicom_instances:
#         instance_id = instance['ID']
#         dicom_response = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/file", stream=True)
#         if dicom_response.status_code == 200:
#             dicom_file = pydicom.dcmread(BytesIO(dicom_response.content))
#             dicom_data.append(dicom_file.pixel_array)

#     # Supposons que nous avons une pile d'images (3D)
#     dicom_array = np.stack(dicom_data, axis=-1)
#     affine = np.eye(4)  # Simplification: l'affine est supposé être l'identité

#     nifti_image = nib.Nifti1Image(dicom_array, affine)
#     nib.save(nifti_image, 'temp_image.nii')  # Sauvegarde temporaire pour utilisation dans le modèle
#     return nifti_image

# def prepare_model_input(dicom_instances):
#     # Convertir les images DICOM en format NIfTI ou directement en tenseurs
#     # (Ici, nous supposons une fonction existante qui gère cette conversion)
#     nifti_image = convert_dicom_to_nifti(dicom_instances)
#     # Convertir NIfTI en tenseurs
#     transform = Compose([LoadImage(image_only=True), ToTensor()])
#     input_tensor = transform(nifti_image.get_fdata())
#     input_tensor = input_tensor.unsqueeze(0)  # Ajouter une dimension batch si nécessaire
#     return input_tensor


# def process_model_output(output):
#     """
#     Convertit les sorties du modèle (supposées être des masques de segmentation pour chaque métastase)
#     en une liste de dictionnaires, chaque dictionnaire contenant les coordonnées des pixels pour une métastase.
    
#     Args:
#         output (np.array): Un tenseur numpy avec des dimensions (num_metastases, height, width),
#                            où chaque 'slice' représente le masque d'une métastase.
    
#     Returns:
#         list of dicts: Une liste de dictionnaires avec des clés 'id' et 'coordinates', où 'coordinates'
#                        est une liste de tuples (x, y) pour chaque pixel appartenant à la métastase.
#     """
#     metastases_info = []
#     num_metastases, height, width = output.shape
#     for i in range(num_metastases):
#         mask = output[i]
#         coordinates = np.argwhere(mask == 1)
#         # Convertir les coordonnées en liste de tuples
#         coordinates_list = [tuple(coord) for coord in coordinates]
#         metastases_info.append({
#             'id': f'GTV{i+1}',
#             'coordinates': coordinates_list
#         })
    
#     return metastases_info


# @app.route('/segmentation/<study_id>', methods=['POST'])
# def segmentation(study_id):
#     # Fetch DICOM data
#     dicom_files = fetch_dicom_files(study_id)
#     if not dicom_files:
#         return jsonify({"error": "No DICOM files found"}), 404

#     # Convert DICOM to NIfTI
#     nifti_image = convert_dicom_to_nifti(dicom_files)

#     # Preprocess and convert NIfTI to tensor
#     input_tensor = preprocess_nifti(nifti_image)

#     # Predict using the model
#     model_output = model(input_tensor)

#     # Process and return the model output
#     return jsonify({"result": model_output.tolist()})

# def fetch_dicom_files(study_id):
#     """Retrieve DICOM files from a PACS server given a study ID."""
#     # Implementation depends on your PACS server API
    
#     return list_of_dicom_files

# def convert_dicom_to_nifti(dicom_files):
#     """Convert a list of DICOM files to a single NIfTI image."""
#     dicom_data = [pydicom.dcmread(BytesIO(f)) for f in dicom_files]
#     dicom_arrays = np.stack([d.pixel_array for d in dicom_data], axis=-1)
#     affine = np.eye(4)  # Simplistic assumption
#     nifti_image = nib.Nifti1Image(dicom_arrays, affine)
#     return nifti_image

# def preprocess_nifti(nifti_image):
#     """Preprocess NIfTI image and convert it to a tensor suitable for the model."""
#     transform = Compose([LoadImage(image_only=True), ToTensor()])
#     input_tensor = transform(nifti_image.get_fdata())
#     input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
#     return input_tensor



def find_orthanc_study_id_by_study_instance_uid(study_instance_uid):
    studies_response = requests.get(f"{ORTHANC_URL}/studies")
    
    if studies_response.status_code == 200:
        print("okokokokokokokok")
        studies = studies_response.json()
        
        for study in studies:
            study_details_response = requests.get(f"{ORTHANC_URL}/studies/{study}")
            
            if study_details_response.status_code == 200:
                study_details = study_details_response.json()
                
                if 'MainDicomTags' in study_details and study_details['MainDicomTags']['StudyInstanceUID'] == study_instance_uid:
                    return study  # L'ID Orthanc de l'étude
            else:
                print(f"Erreur lors de la récupération des détails de l'étude {study}: {study_details_response.status_code}")
                continue
    else:
        print("Erreur lors de la récupération de la liste des études:", studies_response.status_code)
    
    return None
    
"""
def upload_rtstruct(rtstruct_path):
    try:
        print(f"Uploading RTStruct from {rtstruct_path}")
        
        with open(rtstruct_path, 'rb') as f:
            files = {'file': (os.path.basename(rtstruct_path), f, 'application/dicom')}
            response = requests.post(f"{ORTHANC_URL}/instances", files=files)
        
        if response.status_code in [200, 202]:
            orthanc_response = response.json() if response.content else "No JSON content in response"
            return jsonify({"success": "RTStruct uploaded successfully", "OrthancResponse": orthanc_response}), response.status_code
        else:
            return jsonify({"error": "Failed to upload RTStruct to Orthanc", "OrthancResponse": response.text}), response.status_code
    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500
"""

def upload_rtstruct(rtstruct):
    buffer = rtstruct.save_to_memory()  # Récupère le buffer en mémoire contenant le RTStruct
    print("Uploading RTStruct...")
    print(buffer)
    try:
        files = {'file': ('rtstruct.dcm', buffer, 'application/dicom')}
        response = requests.post(f"{ORTHANC_URL}/instances", files=files)

        if response.status_code in [200, 202]:
            orthanc_response = response.json() if response.content else "No JSON content in response"
            return jsonify({"success": "RTStruct uploaded successfully", "OrthancResponse": orthanc_response}), response.status_code
        else:
            return jsonify({"error": "Failed to upload RTStruct to Orthanc", "OrthancResponse": response.text}), response.status_code
    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500
    

def load_dicom_datasets(dicom_folder_path: str) -> List[Dataset]:
    """
    Charge tous les fichiers DICOM d'un dossier donné en datasets pydicom.
    """
    dicom_datasets = []
    for filename in os.listdir(dicom_folder_path):
        if filename.endswith('.dcm'):
            file_path = os.path.join(dicom_folder_path, filename)
            try:
                ds = pydicom.dcmread(file_path)
                dicom_datasets.append(ds)
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier DICOM {filename}: {e}")
                continue  # ou lever une exception selon les besoins de votre application

    if not dicom_datasets:
        raise Exception("Aucun fichier DICOM valide trouvé dans le dossier spécifié.")

    return dicom_datasets
    
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = MesuresDB("./BDD/mesures.db")
    print('DB:', db.connexion)
    return db

@app.teardown_appcontext
def close_db(error):
    db = getattr(g, '_database', None)
    if db is not None:
        db.connexion.close()








if __name__ == '__main__':
    app.run(debug=True)
