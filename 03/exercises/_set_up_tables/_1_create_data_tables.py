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

DATAFILES_DIR = "/app/exercises/subject"

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
            Sensitivity     float8,
            Hability        float8,
            Strength        float8,
            Power           float8,
            Agility         float8,
            Dexterity       float8,
            Awareness       float8,
            Prescience      float8,
            Reactivity      float8,
            Midi_chlorien   float8,
            Slash           float8,
            Push            float8,
            Pull            float8,
            Lightsaber      float8,
            Survival        float8,
            Repulse         float8,
            Friendship      float8,
            Blocking        float8,
            Deflection      float8,
            Mass            float8,
            Recovery        float8,
            Evade           float8,
            Stims           float8,
            Sprint          float8,
            Combo           float8,
            Delay           float8,
            Attunement      float8,
            Empowered       float8,
            Burst           float8,
            Grasping        float8,
            knight          text
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
    file = DATAFILES_DIR + '/Test_knight.csv'
    files = glob.glob(DATAFILES_DIR + "/Test_knight.csv")
    print(files)
    table_name = Path(file).stem
    print (f"\nCreating table: {table_name}")
    print ("-----------------------------")
    create_and_fill_table('train_knight', DATAFILES_DIR + '/Train_knight.csv')

