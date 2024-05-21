import sqlite3
from MesuresSQLite import MesuresDB

def main():
    # Créer une instance de la base de données
    db = MesuresDB('mesures.db')

    # Ajouter des patients fictifs
    db.ajouter_patient("Alice Dupont", "1980-05-15", "F")
    db.ajouter_patient("Bob Martin", "1975-11-23", "M")

    # Ajouter des études pour Alice
    db.ajouter_etude(1, "Serie1", "SOP1", "2021-06-01")
    # Ajouter des métastases à l'étude de Alice
    db.ajouter_metastase(1, 12.5, 2.5, 1, 5)
    db.ajouter_metastase(1, 15.0, 3.0, 6, 10)
    db.ajouter_metastase(1, 10.0, 1.5, 11, 15)

    # Ajouter des études pour Bob
    db.ajouter_etude(2, "Serie2", "SOP2", "2022-02-20")
    db.ajouter_metastase(2, 20.0, 4.0, 1, 4)  # Première étude de Bob
    db.ajouter_etude(2, "Serie3", "SOP3", "2022-03-15")
    db.ajouter_metastase(3, 25.0, 5.0, 2, 7)  # Deuxième étude de Bob
    db.ajouter_metastase(3, 30.0, 6.0, 8, 12)  # Deuxième étude de Bob

if __name__ == "__main__":
    main()
