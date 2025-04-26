up:
	docker compose build
	docker compose up -d --wait

down:
	docker compose down

psql:
	docker exec -it postgres_db psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}
run:
	docker exec -it postgres_db bash

run-pgadmin:
	docker exec -it pgadmin bash

copy-data:
	docker cp postgres_db:/var/lib/postgresql/data ./data

logs:
	docker compose logs -f postgres pgadmin

clean:
	- docker stop $$(docker ps -aq); docker rm $$(docker ps -aq); docker volume rm $$(docker volume ls -q)

remove-all: down clean

.PHONY: up down psql run logs clean remove-all
