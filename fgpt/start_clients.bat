@echo off
echo Starting Flower Clients...

start "Client 0" cmd /k python -m flower_supernode --insecure --superlink="127.0.0.1:8080" --app=client_app:main --app-args="--client_id=0"
timeout /t 10

start "Client 1" cmd /k python -m flower_supernode --insecure --superlink="127.0.0.1:8080" --app=client_app:main --app-args="--client_id=1"
@REM timeout /t 10

@REM start "Client 2" cmd /k python -m flower_supernode --insecure --superlink="127.0.0.1:8080" --app=client_app:main --app-args="--client_id=2"
