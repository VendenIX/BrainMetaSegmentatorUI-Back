import os
import shutil
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from pydicom.filereader import InvalidDicomError
import zipfile
import io
import json
import pydicom
import requests
from flask import Flask, g, jsonify, render_template, request
from flask_cors import CORS
import time
from pydicom.dataset import Dataset

from BDD.MesuresSQLite import MesuresDB
from mock import simulate_rtstruct_generation2
from segmentation import generate_rtstruct_segmentation_unetr, extract_roi_info

app = Flask(__name__)
CORS(app)
ORTHANC_URL = "http://localhost:8042"

#model_path = "C:\MetIA\models\checkpoint-epoch=1599-val_loss=0.225.ckpt"
model_path = "C:\MetIA\models\checkpoint-epoch=2409-val_loss=0.306.ckpt"

#model_path = "C:\MetIA\BrainMetaSegmentatorUI-Back\model\checkpoint-epoch=2409-val_loss=0.306.ckpt"

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
    files = request.files.getlist('files[]')
    print(f"{len(files)} fichiers reçus")

    try:
        dicoms = []
        rtstruct = None

        for file in files:
            print(f"Fichier reçu : {file.filename}")

            if not file.filename.endswith('.dcm'):
                return jsonify({"error": f"Le fichier {file.filename} n'est pas un fichier DICOM (.dcm)"}), 400

            # Lire le fichier DICOM en mémoire
            file_bytes = BytesIO(file.read())
            dicom_data = pydicom.dcmread(file_bytes)

            if dicom_data.Modality == 'RTSTRUCT':
                rtstruct = dicom_data
            else:
                dicoms.append(dicom_data)

            # Réinitialiser le pointeur du fichier pour l'upload
            file.seek(0)

            # Effectuer l'upload vers Orthanc
            upload_files = {'file': (file.filename, file, 'application/dicom')}
            response = requests.post(f"{ORTHANC_URL}/instances", files=upload_files)
            print(f"Réponse d'Orthanc : Statut {response.status_code}, Contenu {response.content}")

        # Si on envoie un RTStruct sans images dicoms avec, on va aller chercher sur le serveur ses images dicoms
        if rtstruct and not dicoms:
            print("est ce que je rentre ici au moins ?")
            study_instance_uid = rtstruct.StudyInstanceUID
            orthanc_study_id = find_orthanc_id_by_sop_instance_uid(study_instance_uid)
            dicoms, _, _ = download_and_process_dicoms(orthanc_study_id)
        print("len :")
        print(len(dicoms))
        # Si on a les dicoms et le rtstruct, on peut ajouter les données des meta à la base de donnée
        if rtstruct and dicoms:
            meta_infos, rtstruct_infos = extract_roi_info(rtstruct, dicoms)
            # Convertir les informations du RTStruct
            patient_id = int(rtstruct_infos["PatientID"])  # Convertir en entier
            patient_name = str(rtstruct_infos["PatientName"])
            patient_birth_date = convertir_date(rtstruct_infos["PatientBirthDate"])
            patient_sex = rtstruct_infos["PatientSex"]
            study_date = convertir_date(rtstruct_infos["StudyDate"])
            study_instance_uid = rtstruct_infos["StudyInstanceUID"]

            # Impression des informations des ROIs
            for roi_name, info in meta_infos.items():
                print(f"{roi_name} : Diamètre max: {info['diameter_max']:.2f} mm, Volume: {info['volume_cm3']:.2f} cm³, Slice de début: {info['start_slice']} ,Slice de fin: {info['end_slice']}")

            db = get_db()

            # Ajout du patient
            db.ajouter_patient(patient_id, patient_name, patient_birth_date, patient_sex)

            # Ajout de l'étude
            db.ajouter_etude(study_instance_uid, patient_id, date_traitement=study_date)

            # Ajout des métastases
            for roi_name, info in meta_infos.items():
                db.ajouter_metastase(study_instance_uid, str(roi_name), info["volume_cm3"], info["diameter_max"], info["start_slice"], info["end_slice"])

        return jsonify({"success": "DICOM files uploaded and processed successfully"}), 200

    except Exception as e:
        print(f"Erreur lors du traitement des fichiers DICOM : {e}")
        return jsonify({"error": "Erreur serveur"}), 500


def download_and_process_dicoms(orthanc_study_id):
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
                            rtstruct_id = find_orthanc_id_by_sop_instance_uid(dicom_data.SOPInstanceUID)
                        else:
                            dicom_datasets.append(dicom_data)
                    except InvalidDicomError:
                        continue  # Gérer ou ignorer les fichiers non DICOM ou corrompus

            print(f"Loaded {len(dicom_datasets)} DICOM files into memory.")
            dicom_datasets.sort(key=lambda x: x.InstanceNumber if 'InstanceNumber' in dir(x) else 0)
            return dicom_datasets, rtstruct, rtstruct_id
        else:
            print("Failed to retrieve study:", response.status)
            return None, None, None
    except requests.RequestException as e:
        print("Error during request:", str(e))
        return None, None, None


"""
Permet de supprimer une étude DICOM du serveur Orthanc et de la base de données locale
"""
@app.route('/delete-study/<study_instance_uid>', methods=['DELETE'])
def delete_study(study_instance_uid):
    orthanc_study_id = find_orthanc_id_by_sop_instance_uid(study_instance_uid)
    print("Je supprime l'instance DICOM avec StudyInstanceUID", study_instance_uid, "et ID Orthanc", orthanc_study_id)

    try:
        # Envoie une requête DELETE à Orthanc
        response = requests.delete(f"{ORTHANC_URL}/studies/{orthanc_study_id}")
        # Vérifie si la suppression a réussi
        if response.status_code == 200:
            db = get_db()
            patient_id = db.get_patient_id_by_study_instance_uid(study_instance_uid)
            db.supprimer_etude(study_instance_uid)
            if not db.patient_has_studies(patient_id):
                print("le patient n'a plus d'études !")
                print(" je vais supprimer le patient numéro ", patient_id)
                db.supprimer_patient(patient_id)
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
    except Exception as e:
        # Gestion des erreurs de base de données ou autres erreurs inattendues
        return jsonify({"error": f"Erreur lors de la suppression de l'étude: {str(e)}"}), 500



"""
Permet de lancer la segmentation d'une étude DICOM spécifique
"""
@app.route('/segmentation/<study_instance_uid>', methods=['POST'])
def segmentation(study_instance_uid):
    # Convertir StudyInstanceUID en ID Orthanc
    orthanc_study_id = find_orthanc_id_by_sop_instance_uid(study_instance_uid)
    if not orthanc_study_id:
        return jsonify({"error": "StudyInstanceUID not found"}), 404

    try:
        # dicom_data, rtstruct_data, rtstruct_id = get_dicom_instances_and_rtstruct(orthanc_study_id)
        dicom_data, rtstruct_data, rtstruct_id = download_and_process_dicoms(orthanc_study_id)
        #rtstruct, isFromCurrentRTStruct = simulate_rtstruct_generation2(dicom_data, rtstruct_data)  # MOCK (FAKE RTSTRUCT)
        rtstruct, isFromCurrentRTStruct = generate_rtstruct_segmentation_unetr(dicom_data, model_path, rtstruct_data)  # MODELE (ATTENTION A LA RAM)

        if isFromCurrentRTStruct:
            print("Voici l'id du RTStruct :", rtstruct_id)
            update_or_upload_rtstruct(dicom_data, rtstruct, rtstruct_id)
        else:
            update_or_upload_rtstruct(dicom_data, rtstruct)

        return jsonify({"success": "DICOM files retrieved and processed successfully."}), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


"""
Permet d'acquérir l'ID Orthanc relative à un StudyInstanceUID
Ça sert pour la récupération des instances d'une étude et pour supprimer un RTStruct d'une étude
"""


def find_orthanc_id_by_sop_instance_uid(study_instance_uid):
    url = f"{ORTHANC_URL}/tools/lookup"
    headers = {'content-type': 'text/plain'}

    # Envoyer le StudyInstanceUID pour la recherche
    response = requests.post(url, data=study_instance_uid, headers=headers)
    if response.status_code == 200:
        result = response.json()
        if result:
            # Supposons que la première correspondance est celle souhaitée
            first_match = result[0]
            return first_match['ID']  # Retourne l'ID Orthanc de l'étude
        else:
            print("No matching study found.")
            return None
    else:
        print("Failed to retrieve study ID:", response.status_code, response.reason)
        return None


"""
Permet d'envoyer un RTStruct vers le serveur Orthanc
params : rtstruct -> le RTStruct à envoyer
"""
def update_or_upload_rtstruct(dicoms, rtstruct, rtstruct_id=None):
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
    rtstruct_pydicom = pydicom.dcmread(BytesIO(buffer.getvalue()))
    meta_infos, rtstruct_infos = extract_roi_info(rtstruct_pydicom, dicoms)

    # Convertir les informations du RTStruct
    patient_id = int(rtstruct_infos["PatientID"])  # Convertir en entier
    patient_name = str(rtstruct_infos["PatientName"])
    patient_birth_date = convertir_date(rtstruct_infos["PatientBirthDate"])
    patient_sex = rtstruct_infos["PatientSex"]
    study_date = convertir_date(rtstruct_infos["StudyDate"])
    study_instance_uid = rtstruct_infos["StudyInstanceUID"]

    # Impression des informations des ROIs
    for roi_name, info in meta_infos.items():
        print(f"{roi_name} : Diamètre max: {info['diameter_max']:.2f} mm, Volume: {info['volume_cm3']:.2f} cm³, Slice de début: {info['start_slice']} ,Slice de fin: {info['end_slice']}")

    db = get_db()

     # Ajout ou mise à jour du patient
    if not db.patient_exists(patient_id):
        db.ajouter_patient(patient_id, patient_name, patient_birth_date, patient_sex)
    
    # Ajout ou mise à jour de l'étude
    if db.etude_exists(study_instance_uid):
        db.supprimer_etude(study_instance_uid)
    db.ajouter_etude(study_instance_uid, patient_id, date_traitement=study_date)

    # Ajout des métastases
    db.supprimer_metastases_from_etude(study_instance_uid)
    for roi_name, info in meta_infos.items():
        db.ajouter_metastase(study_instance_uid, str(roi_name), info["volume_cm3"], info["diameter_max"], info["start_slice"], info["end_slice"])

    if upload_response.status_code in [200, 202]:
        return jsonify({"success": "RTStruct uploaded successfully"}), upload_response.status_code
    else:
        return jsonify({"error": "Failed to upload RTStruct"}), upload_response.status_code

def convertir_date(date):
    """Convertir une date au format YYYYMMDD en YYYY-MM-DD."""
    if len(date) == 8:
        return f"{date[:4]}-{date[4:6]}-{date[6:]}"
    else:
        raise ValueError("La date doit être au format YYYYMMDD")

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
