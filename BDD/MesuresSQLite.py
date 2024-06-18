import sqlite3
import re 

class MesuresDB:

    def __init__(self, path) -> None:
        self.connexion = sqlite3.connect(path)
        self.curseur = self.connexion.cursor()
        self.initialiser()

    ## INITIALISATION
    def initialiser(self):
        # Création de la table patient avec les champs id, nom, date de naissance, et sexe
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS patient (
                idPatient INTEGER PRIMARY KEY,
                nom TEXT,
                dateNaissance DATE,
                sexe CHAR(1) CHECK (sexe IN ('M', 'F'))
            )
        ''')

        # Création de la table étude liée au patient
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS etude (
                StudyInstanceUID TEXT PRIMARY KEY,
                idPatient INTEGER,
                dateTraitement DATE,
                FOREIGN KEY(idPatient) REFERENCES patient(idPatient)
            )
        ''')

        # Création de la table métastase liée à l'étude
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS metastase(
                idMetastase INTEGER PRIMARY KEY AUTOINCREMENT,
                StudyInstanceUID TEXT,
                nom_metastase TEXT,
                volume REAL,
                diametre REAL NULL,
                slideDebut INTEGER,
                slideFin INTEGER,
                FOREIGN KEY(StudyInstanceUID) REFERENCES etude(StudyInstanceUID)
            )
        ''')

        self.connexion.commit()

    ## PATIENT
    def ajouter_patient(self, id_patient, nom, date_naissance, sexe):
        if not isinstance(id_patient, int):
            print("type de id patient :", type(id_patient))
            raise ValueError("id_patient should be an integer")
        if not isinstance(date_naissance, str):
            print("type de date :", type(date_naissance))
            raise ValueError("date_naissance should be a string in the format YYYY-MM-DD")
                # Vérification du format de la date de naissance
        if not isinstance(date_naissance, str) or not re.match(r'^\d{4}-\d{2}-\d{2}$', date_naissance):
            print("type de date :", type(date_naissance))
            raise ValueError("date_naissance should be a string in the format YYYY-MM-DD")
        # Vérification du sexe
        if sexe not in ('M', 'F'):
            print("type de sexe :", type(sexe))
            raise ValueError("sexe should be 'M' or 'F'")

        self.curseur.execute("INSERT INTO patient (idPatient, nom, dateNaissance, sexe) VALUES (?, ?, ?, ?)",
                             (id_patient, nom, date_naissance, sexe))
        self.connexion.commit()

    def get_patients(self):
        self.curseur.execute("SELECT * FROM patient")
        patients = self.curseur.fetchall()
        return [{"id": pat[0], "name": pat[1], "date": pat[2], "sexe": pat[3]} for pat in patients]

    def supprimer_patient(self, id_patient):
        self.curseur.execute("DELETE FROM patient WHERE idPatient=?", (id_patient,))
        self.curseur.execute("DELETE FROM etude WHERE idPatient=?", (id_patient,))
        self.connexion.commit()

    def patient_exists(self, id_patient):
        self.curseur.execute("SELECT 1 FROM patient WHERE idPatient=?", (id_patient,))
        return self.curseur.fetchone() is not None
    
    def patient_has_studies(self, id_patient):
        self.curseur.execute("SELECT 1 FROM etude WHERE idPatient=?", (id_patient,))
        return self.curseur.fetchone() is not None
    
    def get_patient_id_by_study_instance_uid(self, study_instance_uid):
        self.curseur.execute("SELECT idPatient FROM etude WHERE StudyInstanceUID=?", (study_instance_uid,))
        result = self.curseur.fetchone()
        return result[0] if result else None

    ## ÉTUDE
    def ajouter_etude(self, study_instance_uid, id_patient, date_traitement=None):
        self.curseur.execute("INSERT INTO etude (StudyInstanceUID, idPatient, dateTraitement) VALUES (?, ?, ?)",
                             (study_instance_uid, id_patient, date_traitement))
        self.connexion.commit()

    def get_etudes(self):
        self.curseur.execute("SELECT * FROM etude")
        etudes = self.curseur.fetchall()
        return [{"id_study": etu[0], "id": etu[1], "date": etu[2]} for etu in etudes]
    
    def get_etudes_from_patient(self, id_patient):
        self.curseur.execute("SELECT * FROM etude WHERE idPatient = ?", (id_patient,))
        etudes = self.curseur.fetchall()
        return [{"id_study": etu[0], "date": etu[2]} for etu in etudes]

    def supprimer_etude(self, study_instance_uid):
        # Supprimer l'étude et les métastases associées
        self.curseur.execute("DELETE FROM metastase WHERE StudyInstanceUID=?", (study_instance_uid,))
        self.curseur.execute("DELETE FROM etude WHERE StudyInstanceUID=?", (study_instance_uid,))
        self.connexion.commit()

    def etude_exists(self, study_instance_uid):
        self.curseur.execute("SELECT 1 FROM etude WHERE StudyInstanceUID=?", (study_instance_uid,))
        return self.curseur.fetchone() is not None

    ## MÉTASTASE
    def ajouter_metastase(self, study_instance_uid, nom_metastase ,volume, diametre, slide_debut, slide_fin):
        self.curseur.execute("INSERT INTO metastase (StudyInstanceUID, nom_metastase,volume, diametre, slideDebut, slideFin) VALUES (?, ?, ?, ?, ?, ?)",
                             (study_instance_uid, nom_metastase ,volume, diametre, slide_debut, slide_fin))
        self.connexion.commit()

    def get_metastases(self):
        self.curseur.execute("SELECT * FROM metastase")
        metastases = self.curseur.fetchall()
        return [{"idMetastase": met[0], "idEtude": met[1], "nom_metastase":met[2], "Volume": met[3], "Diametre": met[4], "Slide_Debut": met[5], "Slide_Fin": met[6]} for met in metastases]

    def get_metastases_from_etude(self, study_instance_uid):
        self.curseur.execute("SELECT * FROM metastase WHERE StudyInstanceUID = ?", (study_instance_uid,))
        metastases = self.curseur.fetchall()
        return [{"idMetastase": met[0], "nom_metastase": met[2], "volume": met[3], "diametre": met[4], "slice_Debut": met[5], "slice_Fin": met[6]} for met in metastases]

    def supprimer_metastases_from_etude(self, study_instance_uid):
        self.curseur.execute("DELETE FROM metastase WHERE StudyInstanceUID=?", (study_instance_uid,))
        self.connexion.commit()

if __name__ == '__main__':
    db = MesuresDB("./mesures.db")
    db.initialiser()