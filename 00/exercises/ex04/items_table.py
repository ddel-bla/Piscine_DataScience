import os, psycopg2, csv, sys
from pathlib import Path

def get_db_config() -> dict:
	return {
		"dbname": os.environ.get("POSTGRES_DB"),
		"user": os.environ.get("POSTGRES_USER"),
		"password": os.environ.get("POSTGRES_PASSWORD"),
		"host": os.environ.get("POSTGRES_HOST", "localhost"),
		"port": os.environ.get("POSTGRES_PORT", "5432"),
	}

def create_table(conn, table_name):
	create_table_sql = f"""
		CREATE TABLE IF NOT EXISTS {table_name} (
			product_id      int4,
			category_id     numeric(50,0),
			category_code   text,
			brand           text
		)
	"""
	with conn.cursor() as cur:
		print(f"Ejecutando: Crear tabla {table_name}")
		cur.execute(create_table_sql)
	conn.commit()

def fill_table(conn, table_name, path_csv):
	with conn.cursor() as cur:
		print(f"Llenando tabla: [{table_name}] con datos del archivo: [{path_csv}]")
		with open(path_csv, 'r') as file:
			next(file)
			cur.copy_from(file, table_name, sep=',', null='')
	conn.commit()

if __name__ == "__main__":
	try:
		path_csv = '/exercises/data/item/item.csv'
		table_name = Path(path_csv).stem
		db_config = get_db_config()
		with psycopg2.connect(**db_config) as conn:
			create_table(conn, table_name)
			fill_table(conn, table_name, path_csv)
		print(f"Proceso completado con éxito para la tabla {table_name}")
	except Exception as error:
		print(f"Error general en el proceso: {error}")
		sys.exit(1)
