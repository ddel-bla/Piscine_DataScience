
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

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def figure_1():
    df = pd.read_csv("/exercises/data/Test_knight.csv")
    print(df.head())

    fig, axs = plt.subplots(6, 5, figsize=(20,20))
    skills = list(df.columns)
    x, y = 0, 0
    for skill in skills:
        axs[y, x].set_title(str(skill))
        axs[y, x].hist(df[skill], bins=42, color='#7fbf7f')
        x += 1
        if (x == 5):
            x = 0
            y += 1
    fig.tight_layout()
    plt.savefig('./figure_1.png', dpi=150)
    plt.close()
    

def figure_2():
    df = pd.read_csv("/exercises/data/Train_knight.csv")
    print(df.head())
    jedi_df = 

    fig, axs = plt.subplots(6, 6, figsize=(20,20))
    skills = list(df.columns)
    x, y = 0, 0
    for skill in skills:
        if 
        axs[y, x].set_title(str(skill))
        axs[y, x].hist(df[skill], bins=42, color='#7fbf7f')
        x += 1
        if (x == 6):
            x = 0
            y += 1
    fig.tight_layout()
    plt.savefig('./figure_2.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    figure_2()
