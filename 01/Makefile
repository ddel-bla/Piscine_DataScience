up:
	docker compose build
	docker compose up -d --wait

down:
	docker compose down

run:
	docker exec -it postgres_db bash

logs:
	docker compose logs -f postgres pgadmin

clean:
	- docker stop $$(docker ps -aq); docker rm $$(docker ps -aq); docker volume rm $$(docker volume ls -q)

fclean: down clean

.PHONY: up down run logs clean fclean
