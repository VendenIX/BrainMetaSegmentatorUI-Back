# Création d'un environnement virtuel python pour le projet
echo "Attention, ce script va créer un environnement virtuel python pour le projet pouvant être volumieux (environ 8Go)."
pip install virtualenv
virtualenv ProjetIRMStack
source ProjetIRMStack/bin/activate
# Installation des dépendances pour lancer le code
pip install -r requirements.txt
echo "\n\n"
echo "Après l'installation, faites les commandes suivantes:\n"
echo "source ProjetIRMStack/bin/activate"