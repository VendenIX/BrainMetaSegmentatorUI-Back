@echo off

echo Démarrage du serveur web
cd "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Front\Viewers-3.7.0"
start "" "node" server.js

echo Démarrage de NGINX
cd /d "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\nginx"
start "" "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\nginx\nginx.exe" -c "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\nginx\conf\nginx.conf"

echo Démarrage de l'API Flask
cd "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\"
start "" "cmd.exe" /K "C:\ProgramData\Anaconda3\Scripts\activate.bat CORRAU_RESIMET && python C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\api.py


:: Attendre quelques secondes pour s'assurer que les services sont lancés
timeout /t 10 /nobreak

:: Ouvrir le navigateur par défaut sur localhost:3000
start "" "http://localhost:3000"