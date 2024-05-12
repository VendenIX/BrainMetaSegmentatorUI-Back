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
        return [{"ID du patient": pat[0], "Nom": pat[1], "Date de naissance": pat[2], "Sexe": pat[3]} for pat in patients]

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
        return [{"ID de l'étude": etu[0], "ID du patient": etu[1], "ID de la série": etu[2], "ID du SOP": etu[3], "Date de traitement": etu[4]} for etu in etudes]
    
    def get_etudes_from_patient(self, id_patient):
        self.curseur.execute("SELECT * FROM etude WHERE idPatient = ?", (id_patient,))
        etudes = self.curseur.fetchall()
        return [{"ID de l'étude": etu[0], "ID de la série": etu[2], "ID du SOP": etu[3], "Date de traitement": etu[4]} for etu in etudes]

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
        return [{"idMetastase": met[0], "idEtude": met[1], "Volume": met[2], "Diamètre": met[3], "Slide Début": met[4], "Slide Fin": met[5]} for met in metastases]

    def get_metastases_from_etude(self, id_etude):
        self.curseur.execute("SELECT * FROM metastase WHERE idEtude = ?", (id_etude,))
        metastases = self.curseur.fetchall()
        return [{"idMetastase": met[0], "Volume": met[2], "Diamètre": met[3], "Slide Début": met[4], "Slide Fin": met[5]} for met in metastases]

    def supprimer_metastase(self, id_metastase):
        self.curseur.execute("DELETE FROM metastase WHERE idMetastase=?", (id_metastase,))
        self.connexion.commit()