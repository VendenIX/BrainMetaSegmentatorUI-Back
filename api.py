import os

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)
ORTHANC_URL = os.getenv("ORTHANC_URL")
RTSTRUCT_FILE_PATH = os.getenv("RTSTRUCT_FILE_PATH")
# Connexion simulée au serveur PACS (Orthanc dans ce cas)
def connect_to_orthanc():
    try:
        response = requests.get(ORTHANC_URL)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        print(e)
    return False

# Simulation de la récupération de la référence du patient
def get_patient_reference(patient_id):
    # Ici, vous devriez utiliser l'ID du patient pour récupérer des informations spécifiques à partir d'Orthanc
    # Cette fonction est laissée comme un exemple. Vous devrez adapter l'implémentation selon vos besoins.
    return "simulated_reference" if patient_id else None

# Simulation de l'exécution du modèle de deep learning
def run_deep_learning_model(patient_reference):
    return RTSTRUCT_FILE_PATH if patient_reference else None

# Envoi du fichier RTStruct au serveur PACS
def send_rtstruct_to_pacs_server(rtstruct_path):
    try:
        with open(rtstruct_path, 'rb') as f:
            files = {'file': (rtstruct_path, f, 'application/dicom')}
            response = requests.post(f"{ORTHANC_URL}/instances", files=files)
            if response.status_code in [200, 202]:
                return True
    except requests.exceptions.RequestException as e:
        print(e)
    except FileNotFoundError as e:
        print(e)
    return False

@app.route('/generate-rtstruct', methods=['POST'])
def generate_rtstruct():
    patient_id = request.json.get('patient_id')
    
    if not connect_to_orthanc():
        return jsonify({"error": "Failed to connect to PACS server"}), 500
    
    patient_reference = get_patient_reference(patient_id)
    if not patient_reference:
        return jsonify({"error": "Failed to retrieve patient reference"}), 404
    
    rtstruct_path = run_deep_learning_model(patient_reference)
    if not rtstruct_path:
        return jsonify({"error": "Failed to generate RTStruct by deep learning model"}), 500
    
    if not send_rtstruct_to_pacs_server(rtstruct_path):
        return jsonify({"error": "Failed to send RTStruct to PACS server"}), 500
    
    return jsonify({"success": "RTStruct generated and uploaded successfully"}), 200

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

if __name__ == '__main__':
    app.run(debug=True)
