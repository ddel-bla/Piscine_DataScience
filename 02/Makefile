include .env

up:
	docker compose build
	docker compose up -d --wait

down:
	docker compose down

postgres:
	docker exec -it postgres_db bash

python:
	docker exec -it python_dev bash

jupyter:
	@echo "Jupyter Lab running at http://localhost:8888"

# Para ejercicios específicos
ex00:
	docker exec -it python_dev python /app/exercises/ex00/pie.py

ex01:
	docker exec -it python_dev python /app/exercises/ex01/chart.py

ex02:
	docker exec -it python_dev python /app/exercises/ex02/mustache.py

ex03:
	docker exec -it python_dev python /app/exercises/ex03/Building.py

ex04:
	docker exec -it python_dev python /app/exercises/ex04/elbow.py

ex05:
	docker exec -it python_dev python /app/exercises/ex05/Clustering.py

# Utilitarios
logs:
	docker compose logs -f

clean:
	docker compose down
	docker stop $$(docker ps -aq); docker rm $$(docker ps -aq); docker volume rm $$(docker volume ls -q)

.PHONY: up down postgres python run ex00 ex01 ex02 ex03 ex04 ex05 jupyter logs clean help