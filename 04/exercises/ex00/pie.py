
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
        "dbname"    : os.environ.get("POSTGRES_DB"),
        "user"      : os.environ.get("POSTGRES_USER"),
        "password"  : os.environ.get("POSTGRES_PASSWORD"),
        "host"      : os.environ.get("POSTGRES_HOST", "localhost"),
        "port"      : os.environ.get("POSTGRES_PORT", "5432"),
    }

def get_event_types_count() -> dict:
    DB_CONFIG = get_db_config()

    response_data = {}
    @timer_decorator
    def exec_instruction(instruction) -> bool:
        nonlocal response_data
        try:
            cur.execute(instruction)
            response_data = cur.fetchall()
            conn.commit()
            return True
        except psycopg2.Error as error:
            conn.rollback()
            print(f"Error executing instruction: {error}")
            return False
    
    get_event_types_count = f"""
    SELECT event_type, COUNT(*) as type_count FROM customers
    GROUP BY event_type
    ORDER BY type_count DESC;
    """

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                if exec_instruction(get_event_types_count) == False:
                    return None
        return response_data
    except Exception as error:
        print(f"Exception: {error}")
        sys.exit(1)

# Import libraries
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    result = get_event_types_count()
    total_count = sum(count for _, count in result)

    types_count = dict(result)

    types_percent = {}
    for x in types_count:
        types_percent[x] = (types_count[x] * 100) / total_count

    # Creating plot
    # Create pie chart with autopct to show percentages
    patches, texts, autotexts = plt.pie(
        types_percent.values(),
        labels=types_percent.keys(),
        autopct='%.2f%%',          # Show percentages with 1 decimal place
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'},  # Add white borders
        textprops={'fontsize': 12}  # Font size for labels
    )

    plt.title('Exercice 00 : American apple Pie', pad=20, fontsize=14)
    plt.tight_layout()

    plt.savefig('./figure_1.png')
    plt.close()
