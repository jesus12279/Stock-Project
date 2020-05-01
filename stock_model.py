import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahoo_fin.stock_info import get_data, tickers_sp500, tickers_nasdaq, tickers_other, get_quote_table
from scipy.signal import savgol_filter, argrelmin, argrelmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

stock_name = input('Input stock name here:\n')
vol_amount = input('Input avg volume for stock:\n')

def stock_info(name, vol):
    stock = pd.DataFrame(get_data(name))

    start_date = '2009-01-01'

    data09 = stock[stock.index > start_date].copy()

    data09['smooth'] = savgol_filter(data09.adjclose, 9, polyorder=2).copy()

    min_id = 'min_ident'
    max_id = 'max_ident'
    data09[min_id] = 0
    data09[max_id] = 0

    min_ids = argrelmin(data09.smooth.values, order = 23)[0].tolist()
    max_ids = argrelmax(data09.smooth.values, order = 23)[0].tolist()

    data09[min_id].iloc[min_ids] = 1
    data09[max_id].iloc[max_ids] = 1

    return data09.min_ident.plot(figsize=(10,10))

stock_info(stock_name, vol_amount)
