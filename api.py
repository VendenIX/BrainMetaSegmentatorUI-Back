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

@app.route('/uploadRTStruct', methods=['POST'])
def upload_rtstruct():
    try:
        rtstruct_path = simulate_rtstruct_generation()
        print(f"Uploading RTStruct from {rtstruct_path}")
        
        with open(rtstruct_path, 'rb') as f:
            files = {'file': (os.path.basename(rtstruct_path), f, 'application/dicom')}
            response = requests.post(f"{ORTHANC_URL}/instances", files=files)
        
        # Vérifiez si la réponse contient un contenu avant de tenter de décoder le JSON
        if response.status_code in [200, 202]:
            try:
                orthanc_response = response.json()  # Tentez de décoder le JSON uniquement si le contenu est présent
            except ValueError:  # Gère l'absence de contenu JSON
                orthanc_response = "No JSON content in response"
            return jsonify({"success": "RTStruct uploaded successfully", "OrthancResponse": orthanc_response}), response.status_code
        else:
            return jsonify({"error": "Failed to upload RTStruct to Orthanc", "OrthancResponse": response.text}), response.status_code
    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500
    
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
