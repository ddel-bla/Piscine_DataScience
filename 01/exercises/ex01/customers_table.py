import os, glob, psycopg2, csv, sys, re, argparse
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

CUSTOMER_DATA_DIR = "/exercises/data/customer"
ITEM_DATA_DIR = "/exercises/data/item"

def get_db_config() -> dict:
    return {
        "dbname": os.environ.get("POSTGRES_DB"),
        "user": os.environ.get("POSTGRES_USER"),
        "password": os.environ.get("POSTGRES_PASSWORD"),
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
    }

@timer_decorator
def exec_instruction(conn, cur, instruction) -> bool:
    print(f"Ejecutando instrucción: {instruction}")
    try:
        cur.execute(instruction)
        conn.commit()
    except psycopg2.Error as error:
        conn.rollback()
        print(f"Error al ejecutar la instrucción: {error}")
        return False
    return True

@timer_decorator
def fill_table_from_csv(conn, cur, table_name, path_csv) -> bool:
    print(f"Llenando tabla: [{table_name}] con datos del archivo: [{path_csv}]")
    try:
        with open(path_csv, 'r') as file:
            next(file)
            cur.copy_from(file, table_name, sep=',', null='')
        conn.commit()
    except FileNotFoundError:
        print(f"Error: Archivo CSV no encontrado en {path_csv}")
        return False
    except psycopg2.Error as error:
        conn.rollback()
        print(f"Error al llenar la tabla {table_name}: {error}")
        return False
    except Exception as error:
        print(f"Error inesperado: {error}")
        return False
    return True

@timer_decorator
def get_tables(conn, cur):
    instruction = """SELECT table_name FROM information_schema.tables 
                     WHERE table_schema = 'public'"""
    try:
        cur.execute(instruction)
        tables = cur.fetchall()
        table_names = [table[0] for table in tables]
        return table_names
    except psycopg2.Error as error:
        conn.rollback()
        print(f"Error al obtener las tablas: {error}")
        return []

def check_valid_customer_table(table_name):
    pattern = r'^data_202\d_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$'
    return bool(re.fullmatch(pattern, table_name, flags=re.IGNORECASE))

@timer_decorator
def create_customer_tables():
    files = glob.glob(CUSTOMER_DATA_DIR + "/data_202*_*.csv")
    if not files:
        print("No se encontraron archivos de clientes")
        return False
    
    success_count = 0
    DB_CONFIG = get_db_config()
    
    for file in files:
        table_name = Path(file).stem
        print(f"\nCreando tabla: {table_name}")
        print("-----------------------------")
        
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
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    if not exec_instruction(conn, cur, create_table_sql):
                        continue
                    
                    if fill_table_from_csv(conn, cur, table_name, file):
                        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cur.fetchone()[0]
                        print(f"Tabla {table_name} creada con {count} registros")
                        success_count += 1
        
        except Exception as error:
            print(f"Error general al procesar {table_name}: {error}")
    
    print(f"\nSe crearon {success_count} de {len(files)} tablas de clientes")
    return success_count > 0

@timer_decorator
def create_items_table():
    path_csv = f"{ITEM_DATA_DIR}/item.csv"
    table_name = "items"
    
    if not os.path.exists(path_csv):
        print(f"Error: El archivo {path_csv} no existe")
        return False
    
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            product_id      int4,
            category_id     numeric(50,0),
            category_code   text,
            brand           text
        )
    """
    
    try:
        DB_CONFIG = get_db_config()
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                if not exec_instruction(conn, cur, create_table_sql):
                    return False
                
                if fill_table_from_csv(conn, cur, table_name, path_csv):
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cur.fetchone()[0]
                    print(f"Tabla {table_name} creada con {count} registros")
                    return True
    
    except Exception as error:
        print(f"Error general al crear la tabla de items: {error}")
        return False
    
    return False

@timer_decorator
def create_combined_customers_table():
    table_name = "customers"
    DB_CONFIG = get_db_config()
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                check_table_exists = f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table_name}'
                    )
                """
                cur.execute(check_table_exists)
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    print(f"La tabla {table_name} ya existe. Eliminándola...")
                    drop_table = f"DROP TABLE {table_name}"
                    if not exec_instruction(conn, cur, drop_table):
                        print(f"No se pudo eliminar la tabla {table_name}. Abortando.")
                        return False
                
                tables = get_tables(conn, cur)
                if not tables:
                    print("No se encontraron tablas en la base de datos.")
                    return False
                
                table_names = [table for table in tables if check_valid_customer_table(table)]
                if not table_names:
                    print("No se encontraron tablas de clientes para combinar.")
                    return False
                
                print(f"Se encontraron {len(table_names)} tablas para combinar: {', '.join(table_names)}")
                
                create_table_instruction = f"CREATE TABLE {table_name} AS\n"
                for i, table in enumerate(table_names):
                    create_table_instruction += f"SELECT * FROM {table}\n"
                    if i < (len(table_names) - 1):
                        create_table_instruction += "UNION ALL\n"
                
                if not exec_instruction(conn, cur, create_table_instruction):
                    print("Error al crear la tabla combinada. Abortando.")
                    return False
                
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                cur.execute(count_query)
                count = cur.fetchone()[0]
                print(f"La tabla {table_name} ha sido creada exitosamente con {count} registros.")
                
                return True

    except Exception as error:
        print(f"Error general en el proceso: {error}")
        return False

@timer_decorator
def create_indices():
    DB_CONFIG = get_db_config()
    
    indices_config = {
        "customers": [
            ("user_id", "btree"),
            ("product_id", "btree"),
            ("event_time", "btree"),
            ("user_session", "hash")
        ],
        "items": [
            ("product_id", "btree", True),
            ("category_id", "btree"),
            ("category_code", "btree")
        ]
    }
    
    indices_created = 0
    indices_failed = 0
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                tables = get_tables(conn, cur)
                
                for table_name, indices in indices_config.items():
                    if table_name not in tables:
                        print(f"La tabla {table_name} no existe. Saltando creación de índices.")
                        continue
                    
                    print(f"\nCreando índices para la tabla {table_name}:")
                    print("-------------------------------------")
                    
                    check_indices_sql = f"""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = '{table_name}'
                    """
                    cur.execute(check_indices_sql)
                    existing_indices = [idx[0] for idx in cur.fetchall()]
                    
                    for index_config in indices:
                        column = index_config[0]
                        index_type = index_config[1]
                        is_primary = len(index_config) > 2 and index_config[2]
                        
                        index_name = f"idx_{table_name}_{column}"
                        
                        if index_name in existing_indices:
                            print(f"El índice {index_name} ya existe. Saltando.")
                            continue
                        
                        if is_primary:
                            index_sql = f"ALTER TABLE {table_name} ADD PRIMARY KEY ({column})"
                        else:
                            index_sql = f"CREATE INDEX {index_name} ON {table_name} USING {index_type} ({column})"
                        
                        print(f"Creando índice: {index_sql}")
                        if exec_instruction(conn, cur, index_sql):
                            indices_created += 1
                        else:
                            indices_failed += 1
                    
                    analyze_sql = f"ANALYZE {table_name}"
                    exec_instruction(conn, cur, analyze_sql)
                
                print(f"\nCreación de índices completada. Éxitos: {indices_created}, Fallos: {indices_failed}")
                return indices_created > 0

    except Exception as error:
        print(f"Error general al crear índices: {error}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Gestión de tablas en la base de datos")
    parser.add_argument('--action', choices=['all', 'customers', 'items', 'combine', 'indices'], 
                        default='all', help='Acción a realizar')
    args = parser.parse_args()
    
    success = True
    
    if args.action in ['all', 'customers']:
        print("\n=== CREANDO TABLAS DE CLIENTES INDIVIDUALES ===")
        success = create_customer_tables() and success
        
    if args.action in ['all', 'items']:
        print("\n=== CREANDO TABLA DE ITEMS ===")
        success = create_items_table() and success
        
    if args.action in ['all', 'combine']:
        print("\n=== CREANDO TABLA COMBINADA DE CLIENTES ===")
        success = create_combined_customers_table() and success
    
    if args.action in ['all', 'indices']:
        print("\n=== CREANDO ÍNDICES EN LAS TABLAS ===")
        success = create_indices() and success
    
    if success:
        print("\nTodos los procesos se completaron con éxito")
    else:
        print("\nAlguno de los procesos falló. Revise los mensajes de error")
        sys.exit(1)

if __name__ == "__main__":
    main()