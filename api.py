import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from mock import simulate_rtstruct_generation

load_dotenv()

app = Flask(__name__)
CORS(app)
ORTHANC_URL = os.getenv("ORTHANC_URL")
RTSTRUCT_FILE_PATH = os.getenv("RTSTRUCT_FILE_PATH")

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

"""
@app.route('/uploadRTStruct/<study_id>', methods=['POST'])
def upload_rtstruct(study_id):
    try:
        # Simuler la vérification de l'existence d'un RTStruct pour l'étude
        # Cette partie devra être implémentée en fonction de votre logique avec Orthanc
        # Pour l'exemple, on suppose qu'aucun RTStruct n'existe (simulation)
        rtstruct_exists = False  # Changez cette logique selon vos besoins
        
        if rtstruct_exists:
            return jsonify({"error": "RTStruct already exists for this study"}), 409
        
        # Simuler l'upload d'un RTStruct mocké
        rtstruct_path = "/path/to/mock/rtstruct/file.dcm"  # Chemin fictif pour l'exemple
        print(f"Uploading RTStruct for study {study_id} from {rtstruct_path}")
        
        # Logique d'upload à implémenter ici
        
        return jsonify({"success": "RTStruct uploaded successfully"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500
"""

"""
@app.route('/segmentation', methods=['POST'])
def segmentation():
    # Simuler la réception d'images DICOM comme fichier ou comme référence
    dicom_images = request.files.getlist('dicom_images')
    
    # Ici, vous pouvez enregistrer les images DICOM si nécessaire et passer les chemins de fichier
    # Pour le moment, nous passons simplement une liste de noms de fichiers factices
    # à notre fonction simulate_rtstruct_generation
    dicom_image_paths = ['image1.dcm', 'image2.dcm']  # Remplacer par les chemins réels si nécessaire
    
    rtstruct_path = simulate_rtstruct_generation(dicom_image_paths)
    
    if rtstruct_path:
        return jsonify({"rtstruct_path": rtstruct_path}), 200
    else:
        return jsonify({"error": "Failed to generate RTStruct"}), 50
"""
if __name__ == '__main__':
    app.run(debug=True)
