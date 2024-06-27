@echo off

:: Définir les chemins des fichiers PID
set WEB_PID_FILE=..\BrainMetaSegmentatorUI-Front\web_server.pid
set FLASK_PID_FILE=flask_api.pid
set NGINX_PID_FILE=nginx\logs\nginx.pid

:: Vérifier si le serveur web est en cours d'exécution
if exist "%WEB_PID_FILE%" (
    echo OHIF already working
) else (
    echo Starting OHIF Viewer 3.7.0 ...
    cd "..\BrainMetaSegmentatorUI-Front\Viewers-3.7.0"
    start "" /B "cmd.exe" /C "node server.js && echo %%PID%% > %WEB_PID_FILE%"
)

:: Vérifier si NGINX est en cours d'exécution
if exist "%NGINX_PID_FILE%" (
    echo NGINX already working
    echo If NGINX was not launched then in this case delete the nginx.pid file which is located in BrainMetaSegmentatorUI/nginx/logs/
) else (
    echo Starting NGINX ...
    cd /d "..\..\BrainMetaSegmentatorUI-Back\nginx"
    start "" /B "cmd.exe" /C "nginx.exe -c conf/nginx.conf && echo %%PID%% > %NGINX_PID_FILE%"
)

:: Vérifier si l'API Flask est en cours d'exécution
if exist "%FLASK_PID_FILE%" (
    echo API already working
) else (
    echo Starting API on anaconda env : CORRAU-RESIMET ...
    cd "..\..\BrainMetaSegmentatorUI-Back\"
    start "" /B "cmd.exe" /C "C:\ProgramData\Anaconda3\Scripts\activate.bat CORRAU_RESIMET && python api.py && echo %%PID%% > %FLASK_PID_FILE%"
)

:: Ouvrir le navigateur par défaut sur localhost:3000
start "" /B http://localhost/orthanc/ui/app/index.html#/"
start "" /B http://localhost:3000"
echo MetIA is available here on your navigator : http://localhost:3000
echo You can access to the interface of the DICOM Web Server here : http://localhost/orthanc
echo For any questions, answer adress it to romain.andres@etu.unicaen.fr or start an issue here : https://github.com/VendenIX/BrainMetaSegmentatorUI-Front/issues
echo Close this terminal will stop the OHIF Server and the API but not the nginx server