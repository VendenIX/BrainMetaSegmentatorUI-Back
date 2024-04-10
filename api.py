import os

from BDD.MesuresSQLite import MesuresDB
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, jsonify, g, render_template
from flask_cors import CORS

from mock import simulate_rtstruct_generation

load_dotenv()

app = Flask(__name__)
app.static_folder = 'static'

# Chemin vers la base de données SQLite contenant les données des métastases
db_path = '/home/lucaldr/Documents/projetIRM/BrainMetaSegmentatorUI-Back/BDD/mesures.db'

# Fonction pour obtenir la connexion à la base de données
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = MesuresDB(db_path)
    return db

# Route pour fermer la connexion à la base de données
@app.teardown_appcontext
def close_db(exc):
    if hasattr(g, 'db'):
        g.db.connexion.close()

CORS(app)
ORTHANC_URL = os.getenv("ORTHANC_URL")


##### DONNÉES TEST #####

with app.app_context():
    mesures_db = get_db()

    # Vérifier si des études existent déjà dans la base de données
    etudes_existantes = mesures_db.afficher_etudes()
    
    # Ajouter les études uniquement si aucune étude n'existe déjà
    if not etudes_existantes:
        # Ajouter les études
        mesures_db.ajouter_etude(id_patient=1, id_serie='S1', id_sop='SOP1', date_traitement='2023-01-31')
        mesures_db.ajouter_etude(id_patient=2, id_serie='S2', id_sop='SOP2', date_traitement='2023-02-15')

        # Ajouter les sessions pour la première étude
        id_etude_1 = 1
        mesures_db.ajouter_session(id_patient=id_etude_1, date_observation='2023-02-01')
        mesures_db.ajouter_session(id_patient=id_etude_1, date_observation='2023-02-15')

        # Ajouter les sessions pour la deuxième étude
        id_etude_2 = 2
        mesures_db.ajouter_session(id_patient=id_etude_2, date_observation='2023-02-20')
        mesures_db.ajouter_session(id_patient=id_etude_2, date_observation='2023-03-05')

        # Ajouter les métastases pour la première session de la première étude
        id_session_1_etude_1 = 1
        mesures_db.ajouter_metastase(id_session=id_session_1_etude_1, volume=10.5, diametre=5.2, slide_debut=1, slide_fin=10)
        mesures_db.ajouter_metastase(id_session=id_session_1_etude_1, volume=10, diametre=4.7, slide_debut=15, slide_fin=25)

        # Ajouter les métastases pour la deuxième session de la première étude
        id_session_2_etude_1 = 2
        mesures_db.ajouter_metastase(id_session=id_session_2_etude_1, volume=15.2, diametre=7.2, slide_debut=5, slide_fin=15)
        mesures_db.ajouter_metastase(id_session=id_session_2_etude_1, volume=12.5, diametre=6.4, slide_debut=20, slide_fin=30)

        # Ajouter les métastases pour la première session de la deuxième étude
        id_session_1_etude_2 = 3
        mesures_db.ajouter_metastase(id_session=id_session_1_etude_2, volume=8.5, diametre=4.2, slide_debut=2, slide_fin=12)
        mesures_db.ajouter_metastase(id_session=id_session_1_etude_2, volume=7.2, diametre=3.8, slide_debut=18, slide_fin=28)

        # Ajouter les métastases pour la deuxième session de la deuxième étude
        id_session_2_etude_2 = 4
        mesures_db.ajouter_metastase(id_session=id_session_2_etude_2, volume=11.8, diametre=6.1, slide_debut=7, slide_fin=17)
        mesures_db.ajouter_metastase(id_session=id_session_2_etude_2, volume=9.6, diametre=5.5, slide_debut=22, slide_fin=32)

##### ROUTES #####

# ACCUEIL
@app.route('/')
def main():
    return render_template('index.html')

## REQUETES SQLITE

# AFFICHER LES ÉTUDES DE LA BDD
@app.route('/etudes', methods=['GET'])
def afficher_etudes():
    with app.app_context():
        db = get_db()
        etudes = db.afficher_etudes()
    return render_template('etudes.html', etudes=etudes)

# AFFICHER LES SESSIONS DE LA BDD
@app.route('/sessions', methods=['GET'])
def afficher_sessions():
    with app.app_context():
        db = get_db()
        sessions = db.afficher_sessions()
    return render_template('sessions.html', sessions=sessions)

# AFFICHER LES MÉTASTASES DE LA BDD
@app.route('/metastases', methods=['GET'])
def afficher_metastase():
    with app.app_context():
        db = get_db()
        metastases = db.afficher_metastases()
    return render_template('metastases.html', metastases=metastases)

# AFFICHER LES SESSIONS D'UN PATIENT
@app.route('/sessions/<int:id_patient>', methods=['GET'])
def afficher_sessions_patient(id_patient):
    with app.app_context():
        db = get_db()
        sessions_dict = db.afficher_sessions_patient(id_patient)
    return render_template('sessions.html', sessions=sessions_dict)

# AFFICHER LES MÉTASTASES PAR SESSION
@app.route('/metastases/<int:id_session>', methods=['GET'])
def afficher_metastases_session(id_session):
    with app.app_context():
        db = get_db()
        metastases = db.afficher_metastases_session(id_session)
    return render_template('metastases.html', metastases=metastases)

# AFFICHER LA PAGE D'ARBORESCENCE
@app.route('/arborescence')
def arborescence():
    with app.app_context():
        mesures_db = get_db()

        # Récupérer tous les patients de la base de données
        patients = {}

        # Récupérer toutes les études de la base de données
        etudes = mesures_db.afficher_etudes()

        # Récupérer toutes les sessions de la base de données
        sessions = mesures_db.afficher_sessions()

        # Récupérer toutes les métastases de la base de données
        metastases = mesures_db.afficher_metastases()

        # Organiser les études, sessions et métastases par patient
        for etude in etudes:
            patient_id = etude[1]
            if patient_id not in patients:
                patients[patient_id] = {'etudes': [], 'sessions': {}}
            patients[patient_id]['etudes'].append(etude)

        for session_id, session_info in sessions.items():
            patient_id = session_info['idPatient']
            if patient_id in patients:
                if session_id not in patients[patient_id]['sessions']:
                    patients[patient_id]['sessions'][session_id] = {'session': session_info, 'metastases': []}

        for metastase in metastases:
            session_id = metastase[1]
            for patient_id, patient_data in patients.items():
                if session_id in patient_data['sessions']:
                    patient_data['sessions'][session_id]['metastases'].append(metastase)

        return render_template('arborescence.html', patients=patients)


## REQUETES ORTHANC

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

@app.route('/uploadRTStruct', methods=['POST'])
def upload_rtstruct():
    try:
        rtstruct_path = simulate_rtstruct_generation()
        print(f"Uploading RTStruct from {rtstruct_path}")
        with open(rtstruct_path, 'rb') as f:
            files = {'file': (os.path.basename(rtstruct_path), f, 'application/dicom')}
            response = requests.post(f"{ORTHANC_URL}/instances", files=files)
        if response.status_code in [200, 202]:
            # Après un envoi réussi, mettre à jour la base de données
            db = get_db()
            # Mock des données reçues par le modèle d'IA
            # Ici faudra faire une bloucle for et ajouter toutes les métastases
            data_from_ai = {"volume": 10.5, "diametre": 5.2, "slideDebut": 1, "slideFin": 10}
            db.ajouter_metastase(**data_from_ai)
            
            return jsonify({"success": "RTStruct uploaded successfully"}), response.status_code
        else:
            return jsonify({"error": "Failed to upload RTStruct to Orthanc"}), response.status_code
    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500

    
@app.route('/segmentation/<study_instance_uid>', methods=['POST'])
def segmentation(study_instance_uid):
    # Convertir StudyInstanceUID en ID Orthanc
    orthanc_study_id = find_orthanc_study_id_by_study_instance_uid(study_instance_uid)
    
    if not orthanc_study_id:
        return jsonify({"error": "StudyInstanceUID not found"}), 404

    try:
        # Utiliser l'ID Orthanc pour récupérer les instances DICOM pour l'étude spécifiée
        response = requests.get(f"{ORTHANC_URL}/studies/{orthanc_study_id}/instances")
        if response.status_code == 200:
            dicom_instances = response.json()
            print("DICOM Instances for StudyInstanceUID", study_instance_uid, ":", dicom_instances)
            
            # Ici, ajouter logique de traitement des images DICOM récupérées
            # Pour le moment, on simule la génération du RTStruct
            rtstruct_path = simulate_rtstruct_generation()  # mock
            
            return upload_rtstruct_mocked(rtstruct_path)
        else:
            return jsonify({"error": "Failed to retrieve DICOM instances"}), response.status_code
    except requests.exceptions.RequestException as e:
        print(e)
        return jsonify({"error": "Server error"}), 500

    
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
    
def upload_rtstruct_mocked(rtstruct_path):
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


if __name__ == '__main__':
    app.run(debug=True)