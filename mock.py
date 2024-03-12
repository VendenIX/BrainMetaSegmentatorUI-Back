import time as t


def simulate_rtstruct_generation():
    print()
    # Attendre 30 secondes pour simuler le traitement des images DICOM
    print("Simulating RTStruct generation...")
    t.sleep(5)


    # Simulation d'un traitement des images DICOM et génération d'un RTStruct
    # Mettez le chemin absolue de votre RTStruct ici pour simuler l'envoi
    rtstruct_path = '/home/romain/Documents/P_R_O_J_E_T_S/projetIRM/BrainMetaSegmentatorUI-Back/rt_struct.dcm'
    return rtstruct_path