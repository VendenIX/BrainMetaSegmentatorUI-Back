# Bienvenue dans la partie back-end du projet SegmentationUI

Ce projet à pour but de permettre à l'interface OHIF Viewer de lancer un algorithme de deep learning visant à détecter les méta-stases dans le cerveau, basé sur l'architecture unetr. Cette partie du projet est donc une API permettant de communiquer entre le DICOM Web Server Orthanc afin de récupérer/gérer les données médicales au format dicoms, et le modèle de deep learning UNETR fine-tuné pour la segmentation des métastases cérébrales.

## Attention

Ce projet est séparé en deux parties :
- La partie back-end (ce dépôt)
- La partie front-end est accessible [ici](https://github.com/VendenIX/BrainMetaSegmentatorUI-Front).

## Prérequis de Configuration pour lancer le modèle sur le dépôt

| Ressource              | Requis                                                  |
|------------------------|---------------------------------------------------------|
| Mémoire RAM            | 8GB VRAM                                      |
| GPU                    | RTX 3050 Cuda            |


## Pour installer les poids pré-entraînés (nécessaire pour lancer le modèle de deep):

https://drive.google.com/file/d/1kR5QuRAuooYcTNLMnMj80Z9IgSs8jtLO/view

Placer ce fichier de 300 mo dans **unetr/pretrainted_models/**


## Installer les librairies :
```
pip install -r requirements.txt
```
Si vous galérez à installer, installez petit à petit les librairies.
mention spéciale pour l'installation de pytorch : 
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111
```


# Lancer le back-end en mode developpement : 
## Lancer le serveur DICOM Web Orthanc et le proxy nginx :

Se créer un réseau nommé 'pacs' avec docker si ce n'est pas fait : 

```
docker network create pacs
```

Ensuite lancer la pile docker en se positionnant à la racine du dépôt
```
docker-compose -f Orthanc/docker-compose.yml up -d
```

## Lancer l'api flask :
Pour lancer l'api : 

```
python3 api.py
```

# Pour lancer le back-end en mode production : 
## Pour Windows : 
Veillez à placer les dépôts du front et du back dans le même répertoire
```
git checkout deploiementWindows
```

Spécifiez dans le .env le path des poids du modèle UNETR.
exemple : 
```
MODEL_PATH='./models/checkpoint_epoch1599_val_loss0255.cpkt'
```

```
Executer le fichier start_services.bat
# vous pouvez en faire un raccourci sur votre bureau ou un exécutable
```


## Si vous voulez supprimez les données médicales présentes sur votre serveur Orthanc local très rapidement
Clean - up:

```
rm -rf ./Orthanc/orthanc-db/*
```

Sinon vous pouvez supprimer le tout proprement à l'aide de l'interface d'administration présente sur le port localhost:8042, ou alors via ohif sur le front.

## Si un mot de passe vous est demandé, c'est ici:
Login : mapdr Password : changestrongpassword

## Pour eteindre le serveur dicom web, il faut lancer la commande suivante:

```
docker-compose -f Orthanc/docker-compose.yml down
```

Et si accessoirement vous voulez supprimer le réseau pacs : 

```
docker network rm pacs # si vous voulez supprimer le réseau
```

<img src="images_readme/logo_unicaen.png" width="200" height="125" alt="University of Caen Normandy Logo">
<img src="images_readme/baclesse_logo.png" width="140" height="125" alt="Centre François Baclesse">
<img src="images_readme/LogoOHIFViewer.png" width="300" height="125" alt="Open Health Imaging Fundation">
<img src="images_readme/mgh_logo.png" width="110" height="125" alt="Massachusetts General Hospital">
<img src="images_readme/harvard_medical_school_logo.png" width="300" height="125" alt="Harvard Medical School">