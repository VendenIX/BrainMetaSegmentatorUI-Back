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
        # Donc là c'est les infos du patient, un patient peut avoir plusieures études
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS patient (
                idPatient INTEGER PRIMARY KEY,
                nom TEXT,
                dateNaissance DATE,
                sexe CHAR(1) CHECK (sexe IN ('M', 'F'))
            )
        ''')

        # Création de la table étude liée au patient
        # Une étude ça peut avoir plusieurs series de RTStruct (pas de support du RTDose, RTPlan, RTImage, etc.)
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS etude (
                StudyInstanceUID TEXT PRIMARY KEY,
                idPatient INTEGER,
                dateTraitement DATE,
                FOREIGN KEY(idPatient) REFERENCES patient(idPatient)
            )
        ''')

        # Une série de RTStruct peut avoir plusieurs métastases
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS series (
                SeriesInstanceUID TEXT PRIMARY KEY,
                StudyInstanceUID TEXT,
                FOREIGN KEY(StudyInstanceUID) REFERENCES etude(StudyInstanceUID)
            )
        ''')

        # Création de la table métastase liée à l'étude
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS metastase(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roiNumber INTEGER,
                SeriesInstanceUID TEXT,
                nom_metastase TEXT,
                volume REAL,
                diametre REAL NULL,
                slideDebut INTEGER,
                slideFin INTEGER,
                color TEXT,
                FOREIGN KEY(SeriesInstanceUID) REFERENCES series(SeriesInstanceUID)
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
        self.curseur.execute("DELETE FROM series WHERE StudyInstanceUID=?", (study_instance_uid,))
        self.curseur.execute("DELETE FROM metastase WHERE SeriesInstanceUID IN (SELECT SeriesInstanceUID FROM series WHERE StudyInstanceUID=?)", (study_instance_uid,))
        self.curseur.execute("DELETE FROM etude WHERE StudyInstanceUID=?", (study_instance_uid,))
        self.connexion.commit()

    def etude_exists(self, study_instance_uid):
        self.curseur.execute("SELECT 1 FROM etude WHERE StudyInstanceUID=?", (study_instance_uid,))
        return self.curseur.fetchone() is not None
    
    ## SERIES

    def ajouter_serie(self, series_instance_uid, study_instance_uid):
        self.curseur.execute("INSERT INTO series (SeriesInstanceUID, StudyInstanceUID) VALUES (?, ?)",
                             (series_instance_uid, study_instance_uid))
        self.connexion.commit()

    def get_series(self):
        self.curseur.execute("SELECT * FROM series")
        series = self.curseur.fetchall()
        return [{"series_instance_uid": ser[0], "study_instance_uid": ser[1]} for ser in series]
    
    def get_series_from_etude(self, study_instance_uid):
        print("apres je vais là")
        self.curseur.execute("SELECT * FROM series WHERE StudyInstanceUID = ?", (study_instance_uid,))
        series = self.curseur.fetchall()
        print("voici a quoi ressemble ce que je vais renvoyer :")
        print(series)
        print(type(series))
        return [{"series_instance_uid": ser[0], "study_instance_uid": ser[1]} for ser in series]
    
    def supprimer_serie_from_etude(self, study_instance_uid):
        self.curseur.execute("DELETE FROM series WHERE StudyInstanceUID=?", (study_instance_uid,))
        self.connexion.commit()

    def serie_exists(self, series_instance_uid):
        self.curseur.execute("SELECT 1 FROM series WHERE SeriesInstanceUID=?", (series_instance_uid,))
        return self.curseur.fetchone() is not None

    ## MÉTASTASE
    def ajouter_metastase(self, roi_number, series_instance_uid, nom_metastase, volume, diametre, slide_debut, slide_fin, color):
        self.curseur.execute("INSERT INTO metastase (roiNumber, SeriesInstanceUID, nom_metastase, volume, diametre, slideDebut, slideFin, color) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (roi_number, series_instance_uid, nom_metastase, volume, diametre, slide_debut, slide_fin, color))
        self.connexion.commit()

    def get_metastases(self):
        self.curseur.execute("SELECT * FROM metastase")
        metastases = self.curseur.fetchall()
        return [{"id": met[0], "roiNumber": met[1], "series_instance_uid": met[2], "nom_metastase": met[3], "volume": met[4], "diametre": met[5], "slice_debut": met[6], "slice_fin": met[7], "color": met[8]} for met in metastases]

    def get_metastases_from_serie(self, series_instance_uid):
        self.curseur.execute("SELECT * FROM metastase WHERE SeriesInstanceUID = ?", (series_instance_uid,))
        metastases = self.curseur.fetchall()
        return [{"id": met[0], "roiNumber": met[1], "nom_metastase": met[3], "volume": met[4], "diametre": met[5], "slice_debut": met[6], "slice_fin": met[7], "color": met[8]} for met in metastases]

    def supprimer_metastases_from_serie(self, series_instance_uid):
        self.curseur.execute("DELETE FROM metastase WHERE SeriesInstanceUID=?", (series_instance_uid,))
        self.connexion.commit()

    def supprimer_metastase_from_serie(self, series_instance_uid, roi_number):
        self.curseur.execute("DELETE FROM metastase WHERE SeriesInstanceUID=? AND roiNumber=?", (series_instance_uid, roi_number))
        self.connexion.commit()

    def renommer_metastase_from_serie(self, series_instance_uid, roi_number, new_name):
        self.curseur.execute("UPDATE metastase SET nom_metastase = ? WHERE SeriesInstanceUID = ? AND roiNumber = ?", (new_name, series_instance_uid, roi_number))
        self.connexion.commit()

if __name__ == '__main__':
    db = MesuresDB("./mesures.db")
    db.initialiser()