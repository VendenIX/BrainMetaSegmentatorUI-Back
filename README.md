# Bienvenue dans la partie back-end du projet SegmentationUI

Ce projet à pour but de permettre à l'interface OHIF Viewer de lancer un algorithme de deep learning visant à détecter les méta-stases dans le cerveau, basé sur l'architecture unetr. Cette partie du projet est donc une API permettant de communiquer entre le dicom web server Orthanc afin de récupérer/gérer les données médicales au format dicoms, et le modèle de deep learning. A savoir que le modèle est très consommateur en mémoire RAM, et qu'il vous faut des données dicoms afin de pouvoir tester le projet. **Si vous avez des données au format dicoms, nous avons simulé un appel au modèle avec un mock que vous pouvez activer à la ligne 171/172 dans le fichier api.py, il vous suffit alors de commenter/decommenter pour activer ou non le mock ou le vrai modèle.**

## Attention

Ce projet est séparé en deux parties :
- La partie back-end (ce dépôt)
- La partie front-end est accessible [ici](https://github.com/VendenIX/BrainMetaSegmentatorUI-Front).

## Prérequis de Configuration pour lancer le modèle sur le dépôt

| Ressource              | Requis                                                  |
|------------------------|---------------------------------------------------------|
| Mémoire RAM            | 140 GB de RAM minimum                                          |
| GPU                    | Vivement recommandé d'utiliser des GPUs avec conda               |

## Prérequis de Configuration pour lancer le mock (simulation)

| Ressource              | Requis                                                  |
|------------------------|---------------------------------------------------------|
| Mémoire RAM            | 8 GB de RAM minimum                                          |


## Pour installer les poids pré-entraînés (nécessaire pour lancer le modèle de deep):

https://drive.google.com/file/d/1kR5QuRAuooYcTNLMnMj80Z9IgSs8jtLO/view

Placer ce fichier de 300 mo dans **unetr/pretrainted_models/**


## Installer les librairies :
```
pip install -r requirements.txt
```
Si vous galérez à installer, installez petit à petit les librairies.


Comme le projet est encore en cours de développement, nous n'avons pas encore fait la pile docker finale, donc il y aura 2 manipulations à effectuer afin de pouvoir lancer la partie back-end :

## Lancer le serveur Orthanc et le proxy nginx :

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
flask --app api run
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
