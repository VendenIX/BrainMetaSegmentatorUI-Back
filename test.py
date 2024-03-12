import requests

ORTHANC_URL = "http://localhost:8042"

def find_orthanc_study_id_by_study_instance_uid(study_instance_uid):
    studies_response = requests.get(f"{ORTHANC_URL}/studies")
    
    if studies_response.status_code == 200:
        studies = studies_response.json()
        
        for study in studies:
            study_details_response = requests.get(f"{ORTHANC_URL}/studies/{study}")
            
            if study_details_response.status_code == 200:
                study_details = study_details_response.json()
                
                if 'MainDicomTags' in study_details and study_details['MainDicomTags']['StudyInstanceUID'] == study_instance_uid:
                    return study  # L'ID Orthanc de l'étude
            else:
                print(f"Erreur lors de la récupération des détails de l'étude {study}: {study_details_response.status_code}")
                continue
    else:
        print("Erreur lors de la récupération de la liste des études:", studies_response.status_code)
    
    return None

# Exemple d'utilisation
study_instance_uid = "1.3.6.1.4.1.14519.5.2.1.7014.4598.492964872630309412859177308186"
orthanc_study_id = find_orthanc_study_id_by_study_instance_uid(study_instance_uid)

if orthanc_study_id:
    print(f"L'ID Orthanc de l'étude avec StudyInstanceUID {study_instance_uid} est : {orthanc_study_id}")
else:
    print("Étude non trouvée.")
