import os, psycopg2, sys
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

def get_db_config() -> dict:
    return {
        "dbname": os.environ.get("POSTGRES_DB"),
        "user": os.environ.get("POSTGRES_USER"),
        "password": os.environ.get("POSTGRES_PASSWORD"),
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
    }

@timer_decorator
def create_indices(conn, cur, table_name):
    @timer_decorator
    def exec_index_instruction(instruction) -> bool:
        print(f"Executing index instruction: {instruction}")
        try:
            cur.execute(instruction)
            conn.commit()
        except psycopg2.Error as error:
            conn.rollback()
            print(f"Error executing index instruction: {error}")
            return False
        return True
    
    indices = []
    if table_name == "customers":
        indices = [
            ("user_id", "btree"),
            ("product_id", "btree"),
            ("event_time", "btree"),
            ("user_session", "hash"),
            ("category_id", "btree"),
            ("category_code", "btree"),
            ("brand", "btree")
        ]
    elif table_name == "items":
        indices = [
            ("product_id", "btree", True),
            ("category_id", "btree"),
            ("category_code", "btree"),
            ("brand", "btree")
        ]
    else:
        print(f"No hay configuración de índices para la tabla {table_name}")
        return
    
    indices_created = 0
    indices_failed = 0
    
    print(f"\nCreando índices para la tabla {table_name}:")
    print("-------------------------------------")
    
    for index_config in indices:
        column = index_config[0]
        index_type = index_config[1]
        is_primary = len(index_config) > 2 and index_config[2]
        
        if is_primary:
            index_sql = f"ALTER TABLE {table_name} ADD PRIMARY KEY ({column})"
        else:
            index_name = f"idx_{table_name}_{column}"
            index_sql = f"CREATE INDEX {index_name} ON {table_name} USING {index_type} ({column})"
        
        print(f"Creando índice: {index_sql}")
        if exec_index_instruction(index_sql):
            indices_created += 1
        else:
            indices_failed += 1
    
    analyze_sql = f"ANALYZE {table_name}"
    exec_index_instruction(analyze_sql)
    
    print(f"\nCreación de índices completada. Éxitos: {indices_created}, Fallos: {indices_failed}")

@timer_decorator
def join_tables(main_table, other_table):
    DB_CONFIG = get_db_config()

    @timer_decorator
    def exec_instruction(instruction) -> bool:
        print(f"Executing instruction: {instruction}")
        try:
            cur.execute(instruction)
            conn.commit()
        except psycopg2.Error as error:
            conn.rollback()
            print(f"Error executing instruction: {error}")
            return False
        return True
    
    clean_items_temp = f"""
    CREATE TABLE temp_{other_table} AS (
        SELECT * FROM {other_table}
        WHERE product_id      IS NOT NULL
            AND category_id   IS NOT NULL
            AND category_code IS NOT NULL
            AND brand         IS NOT NULL
    );
    """
    
    drop_items = f"""
    DROP TABLE {other_table};
    """
    
    rename_items = f"""
    ALTER TABLE temp_{other_table} RENAME TO {other_table};
    """
    
    create_join_temp = f"""
    CREATE TABLE temp_{main_table} AS (
        SELECT {main_table}.*, {other_table}.category_id, {other_table}.category_code, {other_table}.brand FROM {main_table}
        LEFT JOIN {other_table}
        ON {main_table}.product_id = {other_table}.product_id
    );
    """
    
    drop_main = f"""
    DROP TABLE {main_table};
    """
    
    rename_main = f"""
    ALTER TABLE temp_{main_table} RENAME TO {main_table};
    """

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                if not exec_instruction(clean_items_temp):
                    return
                if not exec_instruction(drop_items):
                    return
                if not exec_instruction(rename_items):
                    return
                
                create_indices(conn, cur, other_table)
                
                if not exec_instruction(create_join_temp):
                    return
                if not exec_instruction(drop_main):
                    return
                if not exec_instruction(rename_main):
                    return
                
                create_indices(conn, cur, main_table)
                
                count_query = f"SELECT COUNT(*) FROM {main_table}"
                cur.execute(count_query)
                count = cur.fetchone()[0]
                print(f"La tabla {main_table} combinada tiene {count} registros")

    except Exception as error:
        print(f"Exception: {error}")
        sys.exit(1)

if __name__ == "__main__":
    join_tables("customers", "items")