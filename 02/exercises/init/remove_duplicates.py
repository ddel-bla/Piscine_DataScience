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

def get_db_config() -> dict :
    return {
        "dbname": os.environ.get("POSTGRES_DB"),
        "user": os.environ.get("POSTGRES_USER"),
        "password": os.environ.get("POSTGRES_PASSWORD"),
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
    }

@timer_decorator
def create_indices(conn, cur, table_name):
    """Crea los índices para la tabla especificada"""
    
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
    
    # Configuración de índices para la tabla customers
    if table_name == "customers":
        indices = [
            ("user_id", "btree"),
            ("product_id", "btree"),
            ("event_time", "btree"),
            ("user_session", "hash")
        ]
    else:
        print(f"No hay configuración de índices para la tabla {table_name}")
        return
    
    indices_created = 0
    indices_failed = 0
    
    print(f"\nCreando índices para la tabla {table_name}:")
    print("-------------------------------------")
    
    for column, index_type in indices:
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
    
    # Modificamos para separar las operaciones en pasos individuales
    create_temp_table = f"""
    CREATE TABLE temp_{table_name} AS
    SELECT DISTINCT ON (event_time, event_type, product_id, price, user_id, user_session) *
    FROM {table_name};
    """
    
    drop_original_table = f"""
    DROP TABLE {table_name};
    """
    
    rename_table = f"""
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
                # Paso 1: Crear tabla temporal con datos únicos
                if not exec_instruction(create_temp_table):
                    return
                
                # Paso 2: Eliminar tabla original
                if not exec_instruction(drop_original_table):
                    return
                
                # Paso 3: Renombrar la tabla temporal
                if not exec_instruction(rename_table):
                    return
                
                # Paso 4: Crear índices ANTES del DELETE para optimizar
                create_indices(conn, cur, table_name)
                
                # Paso 5: Ejecutar el DELETE aprovechando los índices
                if not exec_instruction(remove_duplicates_instruc):
                    return

    except Exception as error:
        print(f"Exception: {error}")
        sys.exit(1)

if __name__ == "__main__":
    remove_duplicates("customers")