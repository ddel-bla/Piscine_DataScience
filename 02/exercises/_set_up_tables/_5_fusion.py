
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

def join_tables(main_table, other_table):
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
    
    clean_items_table = f"""
    CREATE TABLE temp_{other_table} AS (
        SELECT * FROM {other_table}
        WHERE product_id      IS NOT NULL
            AND category_id   IS NOT NULL
            AND category_code IS NOT NULL
            AND brand         IS NOT NULL
    );

    DROP TABLE {other_table};
    ALTER TABLE temp_{other_table} RENAME TO {other_table};
    """

    join_tables = f"""
    CREATE TABLE temp_{main_table} AS (
        SELECT {main_table}.*, {other_table}.category_id, {other_table}.category_code, {other_table}.brand FROM {main_table}
        LEFT JOIN {other_table}
        ON {main_table}.product_id = {other_table}.product_id
    );

    DROP TABLE {main_table};
    ALTER TABLE temp_{main_table} RENAME TO {main_table};
    """

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                if exec_instruction(clean_items_table) == False:
                    return
                if exec_instruction(join_tables) == False:
                    return

    except Exception as error:
        print(f"Exception: {error}")
        sys.exit(1)

if __name__ == "__main__":
    join_tables("customers", "items")

# CREATE TABLE all_data_together AS (
# SELECT new_customer.*, items.category_id, items.category_code, items.brand FROM new_customer
# LEFT JOIN items
# ON new_customer.product_id = items.product_id
# LIMIT 100
# );

# -- Create a new table with distinct rows
# CREATE TABLE customer_distinct AS
# SELECT DISTINCT ON (event_time, event_type, product_id, price, user_id, user_session) *
# FROM customer;

# -- Drop the original and rename the new one
# DROP TABLE customer;
# ALTER TABLE customer_distinct RENAME TO customer;
