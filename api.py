import os
import shutil
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from pydicom.filereader import InvalidDicomError
import zipfile
import io
import pydicom
import requests
from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS
from pydicom.dataset import Dataset

from BDD.MesuresSQLite import MesuresDB
from mock import simulate_rtstruct_generation2
from segmentation import generate_rtstruct_segmentation_unetr

app = Flask(__name__)
CORS(app)
ORTHANC_URL = "http://localhost:8042"
model_path = '/Users/romain/Downloads/Modeles_Pre_Entraines/checkpoint_epoch1599_val_loss0255.cpkt'

"""
Récupère la liste des études DICOM stockées dans le serveur Orthanc
"""
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

"""
Récupère les détails d'une étude DICOM spécifique à partir de son ID Orthanc
params : study_id -> l'ID Orthanc de l'étude
"""
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

"""
Récupère les instances DICOM pour une étude spécifique à partir de son ID Orthanc
"""
def get_dicom_instances_and_rtstruct(study_id: str) -> Tuple[List[pydicom.FileDataset], Optional[pydicom.FileDataset]]:
    dicom_data = []
    rtstruct = None

    response = requests.get(f"{ORTHANC_URL}/studies/{study_id}/instances")
    if response.status_code != 200:
        raise requests.exceptions.RequestException(f"Failed to retrieve DICOM instances: {response.status_code}")

    instances = response.json()
    rtstruct_id = None 
    for instance in instances:
        instance_id = instance['ID']
        dicom_response = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/file", stream=True)
        if dicom_response.status_code == 200:
            dicom_file = pydicom.dcmread(BytesIO(dicom_response.content))
            if dicom_file.Modality == 'RTSTRUCT' and rtstruct is None:
                rtstruct = dicom_file
                rtstruct_id = instance_id 
                print("id du rtstruct :", rtstruct_id)
            else:
                dicom_data.append(dicom_file)

    return dicom_data, rtstruct, rtstruct_id

"""
Permet d'upload un fichier DICOM vers le serveur Orthanc
"""
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
    
def download_and_process_dicoms(study_id, orthanc_study_id):
    query = f"{ORTHANC_URL}/studies/{orthanc_study_id}/archive"
    try:
        response = requests.get(query, verify=False)
        if response.status_code == 200:
            print(f"Retrieved study archive: {orthanc_study_id}")
            zip_content = response.content
            file_like_object = io.BytesIO(zip_content)
            zip_object = zipfile.ZipFile(file_like_object)

            dicom_datasets = []
            rtstruct = None
            rtstruct_id = None

            for zip_info in zip_object.infolist():
                with zip_object.open(zip_info) as file:
                    try:
                        dicom_data = pydicom.dcmread(io.BytesIO(file.read()))
                        # Vérifier si c'est un fichier RTSTRUCT
                        orthanc_id = zip_info.filename
                        if dicom_data.Modality == 'RTSTRUCT':
                            rtstruct = dicom_data
                            rtstruct_id = find_id_from_sop_instanceuid(study_id,dicom_data.SOPInstanceUID)
                            print("rtstruct id  peut etre :",rtstruct_id )
                        else:
                            dicom_datasets.append(dicom_data)
                    except InvalidDicomError:
                        continue  # Gérer ou ignorer les fichiers non DICOM ou corrompus

            print(f"Loaded {len(dicom_datasets)} DICOM files and 1 RTSTRUCT into memory.")
            return dicom_datasets, rtstruct, rtstruct_id
        else:
            print("Failed to retrieve study:", response.status)
            return None, None, None
    except requests.RequestException as e:
        print("Error during request:", str(e))
        return None, None, None
    
"""
Permet de supprimer une étude DICOM du serveur Orthanc
"""
@app.route('/delete-study/<study_instance_uid>', methods=['DELETE'])
def delete_study(study_instance_uid):
    orthanc_study_id = find_orthanc_study_id_by_study_instance_uid(study_instance_uid)
    print("Je supprime l'instance DICOM avec StudyInstanceUID", study_instance_uid, "et ID Orthanc", orthanc_study_id)
    
    try:
        # Envoie une requête DELETE à Orthanc
        response = requests.delete(f"{ORTHANC_URL}/studies/{orthanc_study_id}")
        # Vérifie si la suppression a réussi
        if response.status_code == 200:
            return jsonify({"success": "Instance DICOM supprimée avec succès"}), 200
        else:
            return jsonify({
                "error": "Failed to delete DICOM instance",
                "status_code": response.status_code,
                "response_body": response.text
            }), response.status_code

    except requests.exceptions.RequestException as e:
        # Gestion des erreurs de connexion ou autres erreurs de réseau
        return jsonify({"error": "Erreur lors de la connexion à Orthanc", "exception": str(e)}), 500

"""
Permet de lancer la segmentation d'une étude DICOM spécifique
"""
@app.route('/segmentation/<study_instance_uid>', methods=['POST'])
def segmentation(study_instance_uid):
    # Convertir StudyInstanceUID en ID Orthanc
    orthanc_study_id = find_orthanc_study_id_by_study_instance_uid(study_instance_uid)
    
    if not orthanc_study_id:
        return jsonify({"error": "StudyInstanceUID not found"}), 404

    try:
        #dicom_data, rtstruct_data, rtstruct_id = get_dicom_instances_and_rtstruct(orthanc_study_id)
        dicom_data, rtstruct_data, rtstruct_id = download_and_process_dicoms(orthanc_study_id,orthanc_study_id)
        rtstruct, isFromCurrentRTStruct = simulate_rtstruct_generation2(dicom_data, rtstruct_data)  # MOCK (FAKE RTSTRUCT)
        #rtstruct, isFromCurrentRTStruct = generate_rtstruct_segmentation_unetr(dicom_data, model_path, rtstruct_data)  # MODELE (ATTENTION A LA RAM)

        if isFromCurrentRTStruct:
            print("Voici l'id du RTStruct :", rtstruct_id)
            update_or_upload_rtstruct(rtstruct, rtstruct_id)
        else: 
            update_or_upload_rtstruct(rtstruct)
        
        return jsonify({"success": "DICOM files retrieved and processed successfully."}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

"""
Permet d'acquérir l'ID Orthanc d'une étude DICOM à partir de son StudyInstanceUID
"""
#Faudrait s'en passer, c'est pas opti, mais j'ai pas trouvé comment faire autrement
def find_orthanc_study_id_by_study_instance_uid(study_instance_uid):
    studies_response = requests.get(f"{ORTHANC_URL}/studies")
    
    if studies_response.status_code == 200:
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

def get_series_from_study(study_id):
    response = requests.get(f"{ORTHANC_URL}/studies/{study_id}/series")
    if response.status_code == 200:
        series = response.json()
        return [serie['ID'] for serie in series]
    else:
        print("Erreur lors de la récupération des séries:", response.status_code)
        return []

def get_instances_from_series(series_id):
    response = requests.get(f"{ORTHANC_URL}/series/{series_id}/instances")
    if response.status_code == 200:
        instances = response.json()
        return instances
    else:
        print("Erreur lors de la récupération des instances:", response.status_code)
        return []

"""
Permet d'acquérir l'ID Orthanc d'un RTStruct à partir de son SOPInstanceUID
"""
# Faudrait s'en passer, c'est pas opti, mais j'ai pas trouvé comment faire autrement
def find_id_from_sop_instanceuid(study_id, sop_instance_uid):
    # Récupérer toutes les séries associées à l'étude
    series_ids = get_series_from_study(study_id)
    if not series_ids:
        print("Aucune série trouvée pour l'étude:", study_id)
        return None

    # Itérer sur chaque série pour trouver l'instance avec le SOPInstanceUID donné
    for series_id in series_ids:
        instances = get_instances_from_series(series_id)
        if not instances:
            print("Aucune instance trouvée pour la série:", series_id)
            continue

        # Parcourir les instances pour trouver celle avec le SOPInstanceUID spécifique
        for instance in instances:
            instance_details_response = requests.get(f"{ORTHANC_URL}/instances/{instance['ID']}")
            if instance_details_response.status_code == 200:
                instance_details = instance_details_response.json()
                if 'MainDicomTags' in instance_details and instance_details['MainDicomTags']['SOPInstanceUID'] == sop_instance_uid:
                    return instance['ID']  # Retourner l'ID Orthanc de l'instance
            else:
                print(f"Erreur lors de la récupération des détails de l'instance {instance['ID']}: {instance_details_response.status_code}")
                
    print("SOPInstanceUID non trouvé :", sop_instance_uid)
    return None

"""
Permet d'envoyer un RTStruct vers le serveur Orthanc
params : rtstruct -> le RTStruct à envoyer
"""
def update_or_upload_rtstruct(rtstruct, rtstruct_id=None):
    if rtstruct_id:
        # Suppression de l'ancien RTStruct
        delete_response = requests.delete(f"{ORTHANC_URL}/instances/{rtstruct_id}")
        if delete_response.status_code != 200:
            print(delete_response.status_code)
            print(f"Failed to delete old RTStruct: {delete_response.text}")
            return jsonify({"error": "Failed to delete old RTStruct"}), delete_response.status_code

    # Téléchargement du nouveau ou mis à jour RTStruct
    buffer = rtstruct.save_to_memory()
    files = {'file': ('rtstruct.dcm', buffer, 'application/dicom')}
    upload_response = requests.post(f"{ORTHANC_URL}/instances", files=files)

    if upload_response.status_code in [200, 202]:
        return jsonify({"success": "RTStruct uploaded successfully"}), upload_response.status_code
    else:
        return jsonify({"error": "Failed to upload RTStruct"}), upload_response.status_code

    

"""
Permet de récupérer la base de donnée de suivi des patients
"""
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = MesuresDB("./BDD/mesures.db")
    print('DB:', db.connexion)
    return db

"""
Ferme la connexion à la base de donnée de suivi des patients
"""
@app.teardown_appcontext
def close_db(error):
    db = getattr(g, '_database', None)
    if db is not None:
        db.connexion.close()

"""
Get les études de la base de donnée de suivi des patients
Params : idPatient (optionnel) -> retourne les études pour un patient spécifique
"""
@app.route('/followup-etudes', methods=['GET'])
def get_etudes():
    id_patient = request.args.get('idPatient')
    if id_patient is not None:
        etudes = get_db().get_etudes_from_patient(id_patient)
    else:
        etudes = get_db().get_etudes()
    return jsonify(etudes)

"""
Get les patients de la base de donnée de suivi des patients
"""
@app.route('/followup-patients', methods=['GET'])
def get_patients():
    """Get les patients de la base de données de suivi des patients."""
    patients = get_db().get_patients()
    return jsonify(patients)

"""
Get les métastases de la base de donnée de suivi des patients
Params : idEtude (optionnel) -> retourne les metastases pour une étude spécifique
"""
@app.route('/followup-metastases', methods=['GET'])
def get_metastases():
    id_etude = request.args.get('idEtude')
    if id_etude is not None:
        metastases = get_db().get_metastases_from_etude(id_etude)
    else:
        metastases = get_db().get_metastases()
    return jsonify(metastases)


if __name__ == '__main__':
    app.run(debug=True)
