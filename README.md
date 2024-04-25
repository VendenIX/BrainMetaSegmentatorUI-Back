Faire un fichier .env avec dedans : 

ORTHANC_URL = "http://localhost:8042"


Pour lancer le serveur dicom web : 

```
docker network create pacs
docker-compose -f Orthanc/docker-compose.yml up -d
```

Pour lancer l'api : 

```
python3 api.py
```

Clean - up:

```
rm -rf ./Orthanc/orthanc-db/*
```

Vous pouvez ensuite accéder à l'application via l'url suivante: http://localhost:3000/ et http://localhost:8042

Login : mapdr Password : changestrongpassword

Pour eteindre le serveur dicom web, il faut lancer la commande suivante:

```
docker-compose -f Orthanc/docker-compose.yml down
docker network rm pacs # si vous voulez supprimer le réseau
```
