import sqlite3
import MesuresSQLite

def main():
    # Créer une instance de la base de données
    db = MesuresSQLite.MesuresDB('mesures.db')

    # Ajouter une étude
    db.ajouter_etude(1, 'serie1', 'sop1')

    # Afficher toutes les études
    etudes = db.get_etudes()
    print("Études : ", etudes)

    # Ajouter une session
    db.ajouter_session(1, '2022-01-01')

    # Afficher toutes les sessions
    sessions = db.get_sessions()
    print("Sessions : ", sessions)

    # Ajouter une métastase
    db.ajouter_metastase(1, 1.0, 2.0, 1, 10)

    # Afficher toutes les métastases
    metastases = db.get_metastases()
    print("Métastases : ", metastases)

    # Afficher les sessions d'un patient spécifique
    sessions_patient = db.get_sessions_from_patient(1)
    print("Sessions du patient 1 : ", sessions_patient)

    # Supprimer une métastase
    db.supprimer_metastase(1)
    metastases = db.get_metastases()
    print("Métastases après suppression : ", metastases)

    # Supprimer une session
    db.supprimer_session(1)
    sessions = db.get_sessions()
    print("Sessions après suppression : ", sessions)

    # Supprimer une étude
    db.supprimer_etude(1)
    etudes = db.get_etudes()
    print("Études après suppression : ", etudes)

if __name__ == "__main__":
    main()