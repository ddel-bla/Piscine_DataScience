
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
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


def plot_box(data, title, filename, new_style):

    color_grey = (0, 0, 0, 0.42)
    styles = {
        'notch': False, 'vert': False, 'patch_artist': True,
        'widths': 0.6,
        'whiskerprops': dict(color=color_grey),
        'capprops': dict(color=color_grey),
    }
    styles.update(new_style)

    plt.figure(figsize=(8, 5))
    plt.boxplot(data['price'], **styles)

    ax = plt.gca()
    ax.set_facecolor('lightgray')
    ax.set_yticks([])

    plt.title(title)
    plt.xlabel('Price')
    plt.grid(True, linestyle='-', alpha=0.7, color='white')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def mustache():
    customers_count = f"""
    SELECT product_id, price, user_id, user_session
    FROM customers
    WHERE event_type = 'purchase'
    """

    data = exec_query(customers_count)

    df = pd.DataFrame(data, columns=['product_id', 'price', 'user_id', 'user_session'])
    df['price'] = df['price'].str.replace('$', '').astype(float)
    df.sort_values('price', inplace=True)

    stats = {
        'diff_items_count'  : len(df['price'].drop_duplicates()),
        'sales_count'       : len(df['price']),
        'mean_price'        : round(df['price'].mean(), 2),
        'std'               : round(df['price'].std(), 2),
        'median_price'      : round(df['price'].median(), 2),
        'min_price'         : round(df['price'].min(), 2),
        'max_price'         : round(df['price'].max(), 2),
        'first_quartile'    : round(df['price'].quantile(0.25), 2),
        'second_quartile'   : round(df['price'].quantile(0.50), 2),
        'third_quartile'    : round(df['price'].quantile(0.75), 2),
    }

    print(f"""
        Stats from DataFrame:
-------------------------------------
        diff_items: {stats['diff_items_count']}
        count:      {stats['sales_count']}
        mean:       {stats['mean_price']}
        std:        {stats['std']}
        min:        {stats['min_price']}
        25%:        {stats['first_quartile']}
        50%:        {stats['second_quartile']}
        75%:        {stats['third_quartile']}
        max:        {stats['max_price']}
    """)

    color_grey = (0, 0, 0, 0.42)
    style = {
        'boxprops': dict(facecolor=color_grey, color=color_grey),
        'medianprops': dict(color='green'),
        'flierprops': dict(marker='d', markerfacecolor='black', markersize=5)
    }

    plot_box(df, 'All Data', './figure_1.png', style)

    data_zoom_1 = df[(df['price'] >= 0) & (df['price'] <= 12)]

    new_style = {
        'boxprops': dict(facecolor=(0, 1, 0, 0.42), color=color_grey),
        'medianprops': dict(color=color_grey),
        'flierprops': dict(markersize=0)
    }
    style.update(new_style)
    plot_box(data_zoom_1, 'Zoomed In: Price 0-12', './figure_2.png', style)

    styles = {
        'notch': False, 'vert': False, 'patch_artist': True,
        'widths': 0.6,
        'whiskerprops': dict(color=color_grey),
        'capprops': dict(color=color_grey),
        'boxprops': dict(facecolor=(0, 0, 1, 0.42), color=color_grey),
        'medianprops': dict(color=color_grey),
        'flierprops': dict(markersize=5)
    }

    basket_totals = df.groupby(['user_id','user_session'])['price'].sum().reset_index()

    data_zoom_2 = basket_totals[(basket_totals['price'] >= 28) & (basket_totals['price'] <= 45)]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data_zoom_2['price'], **styles)

    ax = plt.gca()
    ax.set_facecolor('lightgray')
    ax.set_yticks([])

    plt.grid(True, linestyle='-', alpha=0.7, color='white')
    plt.tight_layout()
    plt.savefig('./figure_3.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    mustache()
   
