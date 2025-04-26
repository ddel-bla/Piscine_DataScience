# python3 _1_create_data_tables.py && python3 _2_create_items_table.py && python3 _3_create_customers_table.py && python3 _4_remove_duplicates.py && python3 _5_fusion.py

import os, glob, psycopg2, csv, sys
from pathlib import Path

import time
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

DATAFILES_DIR = "/app/exercises/data/customer"

def get_db_config() -> dict :
    return {
        "dbname": os.environ.get("POSTGRES_DB"),
        "user": os.environ.get("POSTGRES_USER"),
        "password": os.environ.get("POSTGRES_PASSWORD"),
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
    }

def create_and_fill_table(table_name, path_csv):
    DB_CONFIG = get_db_config()

    create_table = f"""
        CREATE TABLE {table_name} (
            event_time      timestamp,
            event_type      text,
            product_id      int4,
            price           money,
            user_id         int8,
            user_session    uuid
        )
        """

    @timer_decorator
    def exec_instruction(instruction):
        print (f"Executing instuction: {instruction}")
        try:
            cur.execute(instruction)
            conn.commit()
        except psycopg2.Error as error:
            conn.rollback()
            print(f"Error executing instruction: {error}")
            return False
        return True

    @timer_decorator
    def fill_table_from_csv(table_name, path_csv) -> bool:
        print(f"Filling table: [{table_name}] with data from file: [{path_csv}]")
        try:
            with open(path_csv, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                cur.copy_from(file, table_name, sep=',', null='')
            conn.commit()
        except FileNotFoundError:
            print(f"Error: CSV file not found at {path_csv}")
            return False
        except psycopg2.Error as error:
            conn.rollback()
            print(f"Error executing instruction: {error}")
            return False
        except Exception as error:
            print(f"Unexpected error: {error}")
            return False
        return True

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                if exec_instruction(create_table) == False:
                    return
                if fill_table_from_csv(table_name, path_csv) == False:
                    return

    except Exception as error:
        print(f"Exception: {error}")
        sys.exit(1)

if __name__ == "__main__":
    print("hola")
    files = glob.glob(DATAFILES_DIR + "/data_202*_*.csv")
    print(files)
    for file in files:
        table_name = Path(file).stem
        print (f"\nCreating table: {table_name}")
        print ("-----------------------------")
        create_and_fill_table(table_name, file)
