import sqlite3

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
                idPatient INTEGER PRIMARY KEY AUTOINCREMENT,
                nom TEXT,
                dateNaissance DATE,
                sexe CHAR(1) CHECK (sexe IN ('M', 'F'))
            )
        ''')

        # Création de la table étude liée au patient
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS etude (
                idEtude INTEGER PRIMARY KEY AUTOINCREMENT,
                idPatient INTEGER,
                idSerie TEXT,
                idSOP TEXT,
                dateTraitement DATE,
                FOREIGN KEY(idPatient) REFERENCES patient(idPatient)
            )
        ''')

        # Création de la table métastase liée à l'étude
        self.curseur.execute('''
            CREATE TABLE IF NOT EXISTS metastase(
                idMetastase INTEGER PRIMARY KEY AUTOINCREMENT,
                idEtude INTEGER,
                volume REAL,
                diametre REAL NULL,
                slideDebut INTEGER,
                slideFin INTEGER,
                FOREIGN KEY(idEtude) REFERENCES etude(idEtude)
            )
        ''')

        self.connexion.commit()

    ## PATIENT
    def ajouter_patient(self, nom, date_naissance, sexe):
        self.curseur.execute("INSERT INTO patient (nom, dateNaissance, sexe) VALUES (?, ?, ?)",
                             (nom, date_naissance, sexe))
        self.connexion.commit()

    def get_patients(self):
        self.curseur.execute("SELECT * FROM patient")
        patients = self.curseur.fetchall()
        return [{"id": pat[0], "name": pat[1], "date": pat[2], "sexe": pat[3]} for pat in patients]

    def supprimer_patient(self, id_patient):
        self.curseur.execute("DELETE FROM patient WHERE idPatient=?", (id_patient,))
        self.curseur.execute("DELETE FROM etude WHERE idPatient=?", (id_patient,))
        self.connexion.commit()

    ## ÉTUDE
    def ajouter_etude(self, id_patient, id_serie, id_sop, date_traitement=None):
        self.curseur.execute("INSERT INTO etude (idPatient, idSerie, idSOP, dateTraitement) VALUES (?, ?, ?, ?)",
                             (id_patient, id_serie, id_sop, date_traitement))
        self.connexion.commit()

    def get_etudes(self):
        self.curseur.execute("SELECT * FROM etude")
        etudes = self.curseur.fetchall()
        return [{"id_study": etu[0], "id": etu[1], "id_serie": etu[2], "id_SOP": etu[3], "date": etu[4]} for etu in etudes]
    
    def get_etudes_from_patient(self, id_patient):
        self.curseur.execute("SELECT * FROM etude WHERE idPatient = ?", (id_patient,))
        etudes = self.curseur.fetchall()
        return [{"id_study": etu[0], "id_serie": etu[2], "id_SOP": etu[3], "date": etu[4]} for etu in etudes]

    def supprimer_etude(self, id_etude):
        self.curseur.execute("DELETE FROM etude WHERE idEtude=?", (id_etude,))
        self.curseur.execute("DELETE FROM metastase WHERE idEtude=?", (id_etude,))
        self.connexion.commit()

    ## MÉTASTASE
    def ajouter_metastase(self, id_etude, volume, diametre, slide_debut, slide_fin):
        self.curseur.execute("INSERT INTO metastase (idEtude, volume, diametre, slideDebut, slideFin) VALUES (?, ?, ?, ?, ?)",
                             (id_etude, volume, diametre, slide_debut, slide_fin))
        self.connexion.commit()

    def get_metastases(self):
        self.curseur.execute("SELECT * FROM metastase")
        metastases = self.curseur.fetchall()
        return [{"idMetastase": met[0], "idEtude": met[1], "Volume": met[2], "Diametre": met[3], "Slide_Debut": met[4], "Slide_Fin": met[5]} for met in metastases]

    def get_metastases_from_etude(self, id_etude):
        self.curseur.execute("SELECT * FROM metastase WHERE idEtude = ?", (id_etude,))
        metastases = self.curseur.fetchall()
        return [{"idMetastase": met[0], "volume": met[2], "diametre": met[3], "slice_Debut": met[4], "slice_Fin": met[5]} for met in metastases]

    def supprimer_metastase(self, id_metastase):
        self.curseur.execute("DELETE FROM metastase WHERE idMetastase=?", (id_metastase,))
        self.connexion.commit()