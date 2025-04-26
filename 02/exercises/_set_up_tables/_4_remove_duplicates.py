
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

def remove_duplicates(table_name):
    DB_CONFIG = get_db_config()

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
    
    first_clean = f"""
    CREATE TABLE temp_{table_name} AS
    SELECT DISTINCT ON (event_time, event_type, product_id, price, user_id, user_session) *
    FROM {table_name};

    DROP TABLE {table_name};
    ALTER TABLE temp_{table_name} RENAME TO {table_name};
    """

    remove_duplicates_instruc = f"""
    DELETE FROM {table_name}
    WHERE (product_id, event_type, event_time) IN (
        SELECT product_id, event_type, event_time
        FROM (
            SELECT 
                product_id,
                event_type,
                event_time,
                LAG(event_time) OVER (
                    PARTITION BY product_id, event_type
                    ORDER BY event_time
                ) AS prev_event_time
            FROM {table_name}
        ) AS subquery
        WHERE event_time - prev_event_time <= INTERVAL '1 second'
    );"""

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                if exec_instruction(first_clean) == False:
                    return
                if exec_instruction(remove_duplicates_instruc) == False:
                    return

    except Exception as error:
        print(f"Exception: {error}")
        sys.exit(1)

if __name__ == "__main__":
    remove_duplicates("customers")
    
    
# -- Create a new table with distinct rows
# CREATE TABLE customer_distinct AS
# SELECT DISTINCT ON (event_time, event_type, product_id, price, user_id, user_session) *
# FROM customer;

# -- Drop the original and rename the new one
# DROP TABLE customer;
# ALTER TABLE customer_distinct RENAME TO customer;