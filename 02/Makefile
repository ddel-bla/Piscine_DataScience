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
	docker logs python_dev
	@echo "Jupyter Lab running at http://localhost:8888"


# Para ejercicios específicos
setup1:
	docker exec -it python_dev python /exercises/init/customers_table.py
setup2:
	docker exec -it python_dev python /exercises/init/remove_duplicates.py
setup3:
	docker exec -it python_dev python /exercises/init/fusion.py

ex00:
	docker exec -it python_dev python /exercises/ex00/pie.py

ex01:
	docker exec -it python_dev python /exercises/ex01/chart.py

ex02:
	docker exec -it python_dev python /exercises/ex02/mustache.py

ex03:
	docker exec -it python_dev python /exercises/ex03/Building.py

ex04:
	docker exec -it python_dev python /exercises/ex04/elbow.py

ex05:
	docker exec -it python_dev python /exercises/ex05/Clustering.py

# Utilitarios
logs:
	docker compose logs -f

clean:
	docker compose down
	docker stop $$(docker ps -aq); docker rm $$(docker ps -aq); docker volume rm $$(docker volume ls -q)

.PHONY: up down postgres python run ex00 ex01 ex02 ex03 ex04 ex05 jupyter logs clean help