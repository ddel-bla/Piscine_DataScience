include .env

up:
	docker compose build
	docker compose up -d --wait

down:
	docker compose down

# Database targets
psql:
	docker exec -it postgres_db psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}

run-postgres:
	docker exec -it postgres_db bash

run-pgadmin:
	docker exec -it pgadmin bash

# Python development targets
python-shell:
	docker exec -it python_dev bash

jupyter:
	@echo "Jupyter Lab running at http://localhost:8888"
	@open http://localhost:8888  # MacOS specific, remove for Linux

run-script:
	docker exec -it python_dev python /app/exercises/$(SCRIPT)

test:
	docker exec -it python_dev pytest /app/exercises/tests

python-build:
	docker compose build python_app

python-logs:
	docker compose logs -f python_app

# Data management
copy-data:
	docker cp postgres_db:/var/lib/postgresql/data ./data

copy-output:
	@mkdir -p ./output
	docker cp python_dev:/app/output/ ./output

# General management
logs:
	docker compose logs -f postgres pgadmin python_app

clean:
	- docker stop $$(docker ps -aq); docker rm $$(docker ps -aq); docker volume rm $$(docker volume ls -q)

remove-all: down clean

# Helper targets
install-deps:
	pip install -r python_app/requirements.txt

help:
	@echo "Development Targets:"
	@echo "  up             - Start all services"
	@echo "  down           - Stop all services"
	@echo "  python-shell   - Access Python container shell"
	@echo "  run-script     - Run a Python script (make run-script SCRIPT=path/to/script.py)"
	@echo "  jupyter        - Open Jupyter Lab in browser"
	@echo "  test           - Run tests"
	@echo "  copy-output    - Copy generated plots/output from container"
	@echo ""
	@echo "Database Targets:"
	@echo "  psql           - Connect to PostgreSQL with psql"
	@echo "  run-postgres   - Access Postgres container shell"
	@echo "  run-pgadmin    - Access pgAdmin container shell"
	@echo ""
	@echo "Utility Targets:"
	@echo "  python-build   - Rebuild Python container"
	@echo "  install-deps   - Install Python dependencies locally"
	@echo "  logs           - View service logs"
	@echo "  clean          - Remove all containers and volumes"
	@echo "  help           - Show this help message"

.PHONY: up down psql run-postgres run-pgadmin python-shell jupyter run-script test python-build \
		copy-data copy-output logs clean remove-all install-deps help