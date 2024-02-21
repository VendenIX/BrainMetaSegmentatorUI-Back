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
        
        # verifif si la réponse contient un contenu avant de tenter de décoder le JSON
        if response.status_code in [200, 202]:
            try:
                orthanc_response = response.json()  # décoder le JSON uniquement si le contenu est présent
            except ValueError:  #si absence de contenu JSON
                orthanc_response = "No JSON content in response"
            return jsonify({"success": "RTStruct uploaded successfully", "OrthancResponse": orthanc_response}), response.status_code
        else:
            return jsonify({"error": "Failed to upload RTStruct to Orthanc", "OrthancResponse": response.text}), response.status_code
    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500
    
@app.route('/segmentation/<study_id>', methods=['POST'])
def segmentation(study_id):
    try:
        # Récupération des instances DICOM pour l'étude spécifiée
        response = requests.get(f"{ORTHANC_URL}/studies/{study_id}/instances")
        if response.status_code == 200:
            dicom_instances = response.json()
            print("DICOM Instances for Study ID", study_id, ":", dicom_instances)
            
            # Ici,ajouter logique de traitement des images DICOM récupérées
            # Pour le moment, on simule la génération du RTStruct
            
            rtstruct_path = simulate_rtstruct_generation()  # mock
            
            return upload_rtstruct_mocked(rtstruct_path)
        else:
            return jsonify({"error": "Failed to retrieve DICOM instances"}), response.status_code
    except requests.exceptions.RequestException as e:
        print(e)
        return jsonify({"error": "Server error"}), 500
    
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