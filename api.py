import os
import shutil
import tempfile
from io import BytesIO
from typing import List

import pydicom
import requests
from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS
from pydicom.dataset import Dataset

from BDD.MesuresSQLite import MesuresDB
from mock import simulate_rtstruct_generation2

app = Flask(__name__)
CORS(app)
ORTHANC_URL = "http://localhost:8042"

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

        return jsonify({"success": "DICOM files retrieved and processed successfully."}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


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