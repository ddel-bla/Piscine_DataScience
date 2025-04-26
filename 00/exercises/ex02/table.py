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
            event_time      timestamptz,
            event_type      text,
            product_id      int4,
            price           money,
            user_id         int8,
            user_session    uuid
        )
    """
    
    try:
        with conn.cursor() as cur:
            print(f"Ejecutando: Crear tabla {table_name}")
            cur.execute(create_table_sql)
            conn.commit()
    except Exception as error:
        print(f"Error al crear la tabla: {error}")
        raise

def fill_table(conn, table_name, path_csv):
    try:
        with conn.cursor() as cur:
            print(f"Llenando tabla: [{table_name}] con datos del archivo: [{path_csv}]")
            with open(path_csv, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                cur.copy_from(file, table_name, sep=',', null='')
            conn.commit()
    except Exception as error:
        print(f"Error al llenar la tabla: {error}")
        raise

if __name__ == "__main__":
    path_csv = '/exercises/data/customer/data_2022_nov.csv'
    table_name = Path(path_csv).stem
    DB_CONFIG = get_db_config()
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            create_table(conn, table_name)
            fill_table(conn, table_name, path_csv)
            print(f"Proceso completado con éxito para la tabla {table_name}")
            
    except Exception as error:
        print(f"Error general en el proceso: {error}")
        sys.exit(1)