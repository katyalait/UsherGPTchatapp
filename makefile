setup:
	pip install pip-tools

compile:
	pip-compile backend/requirements.in -o backend/requirements.txt

sync:
	pip-sync backend/requirements.txt

deploy-frontend:
	npm start

deploy-backend:
	daphne project.asgi:application

redis:
	redis-server