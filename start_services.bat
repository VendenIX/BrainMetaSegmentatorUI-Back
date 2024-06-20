@echo off

:: Définir les chemins des fichiers PID
set WEB_PID_FILE=C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Front\web_server.pid
set FLASK_PID_FILE=C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\flask_api.pid
set NGINX_PID_FILE=C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\nginx\logs\nginx.pid

:: Vérifier si le serveur web est en cours d'exécution
if exist "%WEB_PID_FILE%" (
    echo OHIF est deja lance
) else (
    echo Demarrage de OHIF Viewer 3.7.0
    cd "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Front\Viewers-3.7.0"
    start "" /B "cmd.exe" /C "node server.js && echo %%PID%% > %WEB_PID_FILE%"
)

:: Vérifier si NGINX est en cours d'exécution
if exist "%NGINX_PID_FILE%" (
    echo NGINX est deja en cours d'execution
    echo Si jamais NGINX netait pas lance alors dans ce cas supprimez le fichier nginx.pid qui se situe dans BrainMetaSegmentatorUI/nginx/logs/
) else (
    echo Demarrage de NGINX
    cd /d "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\nginx"
    start "" /B "cmd.exe" /C "nginx.exe -c conf/nginx.conf && echo %%PID%% > %NGINX_PID_FILE%"
)

:: Vérifier si l'API Flask est en cours d'exécution
if exist "%FLASK_PID_FILE%" (
    echo API Flask est deja en cours d'exécution
) else (
    echo Demarrage de l'API Flask sur lenvironnement anaconda CORRAU-RESIMET
    cd "C:\MetIA\IRM-Project\BrainMetaSegmentatorUI-Back\"
    start "" /B "cmd.exe" /C "C:\ProgramData\Anaconda3\Scripts\activate.bat CORRAU_RESIMET && python api.py && echo %%PID%% > %FLASK_PID_FILE%"
)

:: Ouvrir le navigateur par défaut sur localhost:3000
start "" /B http://localhost/orthanc/ui/app/index.html#/"
start "" /B http://localhost:3000"
echo MetIA est disponible a ladresse suivante sur votre navigateur : http://localhost:3000
echo Vous pouvez egalement acceder a linterface dadministration du DICOM Web Server a ladresse suivante : http://localhost/orthanc
echo Pour toute question, addressez vous a romain.andres@etu.unicaen.fr ou laissez une issue sur https://github.com/VendenIX/BrainMetaSegmentatorUI-Front/issues
echo Ferme ce terminal eteindra le serveur web et l'API python