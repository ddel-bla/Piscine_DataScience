import os, glob, psycopg2, csv, sys
from pathlib import Path

DATAFILES_DIR = "/exercises/data/customer"

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
			event_time      timestamptz,
			event_type      text,
			product_id      int4,
			price           money,
			user_id         int8,
			user_session    uuid
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

def process_files(file_list, conn):
	if not file_list:
		return
	path_csv = file_list[0]
	table_name = Path(path_csv).stem
	create_table(conn, table_name)
	fill_table(conn, table_name, path_csv)
	process_files(file_list[1:], conn)

if __name__ == "__main__":
	try:
		files = glob.glob(DATAFILES_DIR + "/data_*.csv")
		db_config = get_db_config()
		with psycopg2.connect(**db_config) as conn:
			process_files(files, conn)
		print("Proceso completado con éxito")
	except Exception as error:
		print(f"Error general en el proceso: {error}")
		sys.exit(1)
