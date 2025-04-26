
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

def exec_query(query) -> dict:
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
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                if exec_instruction(query) == False:
                    return None
        conn.close()
        cur.close()
        return response_data
    except Exception as error:
        print(f"Exception: {error}")
        sys.exit(1)

# Import libraries
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

def elbow():
    customers_count = f"""
    SELECT 
        spending_range, 
        SUM(number_clients) AS group_total
    FROM (
        SELECT 
            user_id,
            CASE
                WHEN SUM(price::numeric) < 25 THEN 0
                WHEN SUM(price::numeric) < 75 THEN 50
                WHEN SUM(price::numeric) < 125 THEN 100
                WHEN SUM(price::numeric) < 175 THEN 150
                WHEN SUM(price::numeric) < 225 THEN 200
                ELSE 300
            END AS spending_range,
            COUNT(DISTINCT(user_id)) AS number_clients
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY user_id
    ) AS user_spending
    GROUP BY spending_range
    ORDER BY spending_range
    """
    # LIMIT 10000

    data = exec_query(customers_count)

    df = pd.DataFrame(data, columns=['spending_range', 'group_total'])
    print(df)

    plt.figure(figsize=(10, 6))
    plt.gcf().set_facecolor('lightgray')
    plt.xlabel('monetary value in ₳')
    plt.ylabel('customers')
    plt.grid(axis='both', linestyle='--', alpha=0.7, color='#b6c5d8')
    plt.tight_layout()

    data_zoom = df[(df['spending_range'] >= 0) & (df['spending_range'] < 250)]
    bars = plt.bar(data_zoom['spending_range'], data_zoom['group_total'],
                   color='#b6c5d8',
                   width=49, align='center'
    )
    ax = plt.gca()
    ax.set_facecolor('lightblue')

    plt.savefig('./figure_2.png', dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    elbow()
