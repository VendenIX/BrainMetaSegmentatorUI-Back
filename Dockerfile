# Utilisation d'une image de base Python
FROM python:3.10.12

# Définition du répertoire de travail dans le conteneur
WORKDIR /App

# Copie du fichier de dépendances et installation des dépendances
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copie du dossier App dans le conteneur
COPY App/ ./App

# Définition de la variable d'environnement pour Flask
ENV FLASK_APP=App/BrainMetaSegmentatorUI.py 

# Exposition du port (généralement le port 5000 pour Flask)
EXPOSE 5000

# Définition de la commande pour exécuter l'application Flask
CMD ["flask", "run", "--host=0.0.0.0"]
