
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


def create_first_chart():
    customers_count = f"""
    SELECT 
        CAST(event_time AS DATE) AS date_text,
        COUNT(DISTINCT user_id) AS customers_count
    FROM customers
    WHERE event_type='purchase' and (event_time >= '2022-10-01 00:00:00' and event_time < '2023-3-01 00:00:00')
    GROUP BY date_text
    """

    data = exec_query(customers_count)
    df = pd.DataFrame(data, columns=['date', 'count'])
    df.sort_values('date')

    plt.figure(figsize=(8, 4))
    plt.gcf().set_facecolor('lightgray')

    plt.plot(df['date'], df['count'],
        linestyle='-', 
        linewidth=2, 
        color='tab:blue',
    )

    ax = plt.gca()
    ax.set_facecolor('lightblue')
    ax.xaxis.set_major_locator(MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(DateFormatter('%b:%y'))  # Format dates
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.spines[['top', 'right']].set_visible(False)

    plt.title('Exercice 01 : initial data exploration', fontsize=14, pad=20)
    plt.ylabel('Number of customers')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('./figure_1.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_second_chart():
    customers_count = f"""
    SELECT
        TO_CHAR(event_time, 'YYYY-MM') as month_year,
        SUM(price) AS sales
    FROM customers
    WHERE event_type='purchase' and (event_time >= '2022-10-01 00:00:00' and event_time < '2023-3-01 00:00:00')
    GROUP BY month_year
    """

    data = exec_query(customers_count)

    df = pd.DataFrame(data, columns=['month', 'sales'])
    df['sales'] = df['sales'].str.replace('[$,]', '', regex=True).astype(float).div(1000000)
    df['month'] = pd.to_datetime(df['month'] + '-01')

    plt.figure(figsize=(10, 6))
    plt.gcf().set_facecolor('lightgray')

    bars = plt.bar(df['month'], df['sales'],
                   color='#b6c5d8',
                   width=25
    )

    ax = plt.gca()
    ax.set_facecolor('lightblue')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.1f}'))
    ax.xaxis.set_major_formatter(DateFormatter('%b:%y'))
    ax.spines[['top', 'right']].set_visible(False)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{height:,.2f}M ₳',
                ha='center',
                va='bottom')

    plt.title('Exercice 01 : initial data exploration', fontsize=14, pad=20)
    plt.ylabel('Total Sales in millions of ₳', labelpad=10)
    plt.xlabel('Month', labelpad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('./figure_2.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_third_chart():
    avrg_sales = f"""
    SELECT
        CAST(daily_user_totals.event_date AS DATE) AS date_text,
        AVG(user_total::numeric) AS avg_spent_per_user
    FROM (
        SELECT
            CAST(event_time AS DATE) AS event_date,
            user_id,
            SUM(price) AS user_total
        FROM customers
        WHERE event_type='purchase' and (event_time >= '2022-10-01 00:00:00' and event_time < '2023-3-01 00:00:00')
        GROUP BY event_date, user_id
    ) AS daily_user_totals
    GROUP BY date_text
    """

    data = exec_query(avrg_sales)

    df = pd.DataFrame(data, columns=['month', 'sales'])
    df.sort_values('month')

    plt.figure(figsize=(10, 6))
    plt.gcf().set_facecolor('lightgray')

    plt.plot(df['month'], df['sales'],
        linestyle='-', 
        linewidth=2, 
        color='tab:blue',
    )

    plt.fill_between(df['month'], df['sales'], color='tab:blue', alpha=0.5)

    ax = plt.gca()
    ax.set_facecolor('lightblue')
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_locator(MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(DateFormatter('%b:%y'))
    ax.set_ylim(bottom=0)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.spines[['top', 'right']].set_visible(False)

    plt.title('Exercice 01 : initial data exploration', fontsize=14, pad=20)
    plt.ylabel('average spend/customer in ₳', labelpad=10)
    plt.xlabel('Month', labelpad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('./figure_3.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_first_chart()
    create_second_chart()
    create_third_chart()

    
