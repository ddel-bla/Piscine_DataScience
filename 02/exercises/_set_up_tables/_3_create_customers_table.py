
import os, glob, psycopg2, csv, sys, re
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

def get_db_config() -> dict :
    return {
        "dbname": os.environ.get("POSTGRES_DB"),
        "user": os.environ.get("POSTGRES_USER"),
        "password": os.environ.get("POSTGRES_PASSWORD"),
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
    }

def create_and_join_tables(table_name):
    DB_CONFIG = get_db_config()

    def get_tables():
        instruction = """SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"""
        try:
            cur.execute(instruction)
            tables = cur.fetchall()
            table_names = []
            for table in tables:
                table_names.append(table[0])
            return table_names
        except psycopg2.Error as error:
            conn.rollback()
            print(f"Error executing instruction: {error}")

    @timer_decorator
    def exec_instruction(instruction) -> bool:
        print (f"Executing instuction: {instruction}")
        try:
            cur.execute(instruction)
            conn.commit()
        except psycopg2.Error as error:
            conn.rollback()
            print(f"Error executing instruction: {error}")
            return False
        return True

    def check_valid_table(table_name):
        pattern = r'^data_202\d_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$'
        return bool(re.fullmatch(pattern, table_name, flags=re.IGNORECASE))

    create_table_instruction = f"""CREATE TABLE {table_name} AS"""

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                tables = get_tables()
                table_names = [table for table in tables if check_valid_table(table)]
                for i, table in enumerate(table_names):
                    create_table_instruction += f"\nSELECT * FROM {table}\n"
                    if i < (len(table_names) - 1):
                        create_table_instruction += "UNION ALL"
                if exec_instruction(create_table_instruction) == False:
                    return

    except Exception as error:
        print(f"Exception: {error}")
        sys.exit(1)

if __name__ == "__main__":
    create_and_join_tables("customers")