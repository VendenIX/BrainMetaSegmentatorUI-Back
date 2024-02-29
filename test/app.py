from flask import Flask, render_template, request, jsonify, g
from BDD.MesuresSQLite import MesuresDB
import requests

app = Flask(__name__)

# URL de la base de données Orthanc qui contient les .dcm
orthanc_url = 'http://localhost:8042'

# Chemin vers la base de données SQLite contenant les données des métastases
db_path = '/home/lucaldr/Documents/projetIRM/test/BDD/mesures.sqlite'

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


@app.route('/upload', methods=['GET', 'POST', 'DELETE'])
def upload():

    if request.method == 'POST' :
        rtsruct_path = '/home/lucaldr/Documents/projetIRM/test/data/1.dcm'
        upload_url = orthanc_url + '/instances'
        files = {'file': open(rtsruct_path, 'rb')}
        response = requests.post(upload_url, files=files)
        print(response.text, response.status_code)
        
        if response.status_code == 200 :
            print(jsonify({'message': 'RTStruct uploaded successfully'}))
        else : 
            print(jsonify({'error': 'Failed to upload RTStruct'}))
        return render_template('index.html')
    
    elif request.method == 'GET' :
        instances_url = f'{orthanc_url}/instances'
        response = requests.get(instances_url)
        if response.status_code == 200:
            instances = response.json()
            return render_template('instances.html', instances=instances)
        else:
            return jsonify({'error': 'Failed to fetch instances'}), response.status_code

    
    elif request.method == 'DELETE':

        instance_id = request.args.get('instance_id')  # Identifie l'instance à supprimer
        delete_url = f'{orthanc_url}/instances/{instance_id}'
        response = requests.delete(delete_url)

        if response.status_code == 200:
            return jsonify({'message': 'Instance deleted successfully'}), 200
        else:
            return jsonify({'error': 'Failed to delete instance'}), response.status_code

if __name__ == '__main__' :
    
    app.run(debug=True)