import sqlite3

class MesuresDB:

    def __init__(self, path) -> None:
        self.connexion = sqlite3.connect(path)
        self.curseur = self.connexion.cursor()
        self.initialiser()

    ## INITIALISATION
    def initialiser(self):
        self.curseur.execute('''CREATE TABLE IF NOT EXISTS etude (
                            idEtude INTEGER PRIMARY KEY AUTOINCREMENT,
                            idPatient INTEGER,
                            idSerie TEXT,
                            idSOP TEXT,
                            dateTraitement DATE
                            )''')

        self.curseur.execute('''CREATE TABLE IF NOT EXISTS session (
                            idSession INTEGER PRIMARY KEY AUTOINCREMENT,
                            idPatient INTEGER,
                            dateObservation DATE,
                            FOREIGN KEY(idPatient) REFERENCES etude(idPatient)
                            )''')

        self.curseur.execute('''CREATE TABLE IF NOT EXISTS metastase(
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            idSession INTEGER,
                            volume REAL,
                            diametre REAL NULL,
                            slideDebut INTEGER,
                            slideFin INTEGER,
                            FOREIGN KEY(idSession) REFERENCES session(idSession)
                            )''')

        self.connexion.commit()


    ## MÉTASTASE
    def ajouter_metastase(self, id_session, volume, diametre, slide_debut, slide_fin):
        self.curseur.execute("INSERT INTO metastase (idSession, volume, diametre, slideDebut, slideFin) VALUES (?, ?, ?, ?, ?)",
                             (id_session, volume, diametre, slide_debut, slide_fin))
        self.connexion.commit()

    def afficher_metastases(self):
        self.curseur.execute("SELECT * FROM metastase")
        metastases = self.curseur.fetchall()
        for metastase in metastases:
            print("ID:", metastase[0])
            print("ID de la session:", metastase[1])
            print("Volume:", metastase[2])
            print("Diamètre:", metastase[3])
            print("Slide début:", metastase[4])
            print("Slide fin:", metastase[5])
            print("------------------------")
        return metastases

    def supprimer_metastase(self, id_metastase):
        self.curseur.execute("DELETE FROM metastase WHERE id=?", (id_metastase,))
        self.connexion.commit()

    ## ÉTUDE
    def ajouter_etude(self, id_patient, id_serie, id_sop, date_traitement=None):
        self.curseur.execute("INSERT INTO etude (idPatient, idSerie, idSOP, dateTraitement) VALUES (?, ?, ?, ?)",
                             (id_patient, id_serie, id_sop, date_traitement))
        self.connexion.commit()

    def afficher_etudes(self):
        self.curseur.execute("SELECT * FROM etude")
        etudes = self.curseur.fetchall()
        for etude in etudes:
            print("ID de l'étude:", etude[0])
            print("ID du patient:", etude[1])
            print("ID de la série:", etude[2])
            print("ID du SOP:", etude[3])
            print("Date de traitement:", etude[4])
            print("------------------------")
        return etudes

    def supprimer_etude(self, id_etude):
        self.curseur.execute("DELETE FROM etude WHERE idEtude=?", (id_etude,))
        self.connexion.commit()

    ## SESSION
    def ajouter_session(self, id_patient, date_observation):
        self.curseur.execute("INSERT INTO session (idPatient, dateObservation) VALUES (?, ?)", (id_patient, date_observation))
        self.connexion.commit()

#    def afficher_sessions(self):
#        self.curseur.execute("SELECT * FROM session")
#        sessions = self.curseur.fetchall()
#        for session in sessions:
#            print("ID de la session:", session[0])
#            print("ID du patient:", session[1])
#            print("Date d'observation:", session[2])
#            print("------------------------")
#        return sessions
    
    def afficher_sessions(self):
        self.curseur.execute("SELECT * FROM session")
        sessions = self.curseur.fetchall()
        sessions_dict = {session[0]: {'idPatient': session[1], 'dateObservation': session[2]} for session in sessions}
        return sessions_dict


    def supprimer_session(self, id_session):
        self.curseur.execute("DELETE FROM session WHERE idSession=?", (id_session,))
        self.curseur.execute("DELETE FROM metastase WHERE idSession=?", (id_session,))
        self.connexion.commit()

    ## MÉTHODES SPÉ
        
#    def afficher_sessions_patient(self, id_patient):
#        self.curseur.execute("SELECT * FROM session WHERE idPatient=?", (id_patient,))
#        sessions = self.curseur.fetchall()
#        for session in sessions:
#            print("ID de la session:", session[0])
#            print("ID du patient:", session[1])
#            print("Date d'observation:", session[2])
#            print("------------------------")
#        return sessions
#    
#    def afficher_metastases_session(self, id_session):
#        self.curseur.execute("SELECT * FROM metastase WHERE idSession=?", (id_session,))
#        metastases = self.curseur.fetchall()
#        for metastase in metastases:
#            print("ID de la métastase:", metastase[0])
#            print("ID de la session:", metastase[1])
#            print("Volume:", metastase[2])
#            print("Diamètre:", metastase[3])
#            print("Slide début:", metastase[4])
#            print("Slide fin:", metastase[5])
#            print("------------------------")
#        return metastases

    def afficher_sessions_patient(self, id_patient):
        self.curseur.execute("SELECT * FROM session WHERE idPatient=?", (id_patient,))
        sessions = self.curseur.fetchall()
        sessions_dict = {session[0]: {'idPatient': session[1], 'dateObservation': session[2]} for session in sessions}
        print(sessions_dict)
        return sessions_dict

    def afficher_metastases_session(self, id_session):
        self.curseur.execute("SELECT * FROM metastase WHERE idSession=?", (id_session,))
        metastases = self.curseur.fetchall()
        return metastases