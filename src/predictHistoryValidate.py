import sys
from binance.client import Client
from binance.enums import *
import pandas as pd
import ta
import datetime
import matplotlib.pyplot as plt
import numpy as np
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator, PercentagePriceOscillator
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import requests
from dotenv import load_dotenv
from engineBTCLive import GetDataBTC
from engineIndicatorsLive import IndicatorsLive
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Programa para Descargar los datos, calcular los indicadores y realizar la prediccion, todo en uno para poder validar datos.

def main():
    if len(sys.argv) < 3:
        print("ERROR: [ Parametros necesarios: <interval> <live?> [limit] ]: Por ejemplo: " + sys.argv[0] + " 5m " + "True " + "500 (limit solo se usa para limitar datos live) ")
        sys.exit(1) 
    btc = GetDataBTC()
    interval = sys.argv[1] # '5m'
    live = sys.argv[2].lower() == 'true'
    limit = 1000 # binance te limita a 1000 en live.
    windowrs = 20
    look_ahead_intervals = 4
    pct = 0.5
    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
    df = btc.getBTCData(interval, live, limit)
    #################################################################
    print(df[['Time', 'Close']].tail(3))
    bot = IndicatorsLive()
    #df[['Time', 'Close']].to_csv(('xxxxxx' + interval + '_up02pct.csv'), index=False)
    bot.getIndicatorsCalculated(interval, windowrs, look_ahead_intervals, pct)
    df = pd.read_csv('../data/live/livewithindicators5m_up02pct.csv')
    # Cargar el modelo
    features = [
                'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume',
                'Price_to_EMA_20_Ratio', 'Price_to_EMA_50_Ratio',
                'MACD_line', 'MACD_signal', 'MACD_diff',
                'RSI', 'RSI_overbought', 'RSI_oversold', 'RSI_change',
                'price_to_bbh_ratio', 'price_to_bbm_ratio', 'price_to_bbl_ratio', 'ADX', 'Volume_MA10',
                'Log_Volume', 'dist_to_bbh_pct', 'dist_to_bbl_pct',
                'dist_to_support_pct', 'dist_to_resistance_pct',
                'fng_number', 'Dist_to_EMA_20_pct', 'Dist_to_EMA_50_pct',
                'EMA_5min_diff_pct', 'EMA_30min_diff_pct', 'EMA_1h_diff_pct', 'EMA_6h_diff_pct', 'EMA_24h_diff_pct',
                'Close_5min_pct_change', 'Close_30min_pct_change', 'Close_1h_pct_change', 'Close_6h_pct_change', 'Close_24h_pct_change'
            ]

    try:
        # Cargar el scaler para predecir
        scaler = joblib.load('./models/modeloNeural5m_up02pct_scaled.pkl')
        df.loc[:, features] = scaler.transform(df[features])
    except Exception as e:
        print(f"Error al cargar el scaler o al transformar los datos: {e}")
        
    model = load_model('./models/modeloNeural5m_up02pct_scaled.keras') 
    # Hacer la prediccion
    prediction = model.predict(df.tail(3000)) 
    print("La prediccion es:", prediction)
    # solo para validar datos
    dfval = pd.read_csv('xxxvalidate.csv')
    dfval['Time'] = pd.to_datetime(dfval['Time'])
    dfval['Time'] = dfval['Time'] + pd.Timedelta(hours=1)
    df2 = dfval[['Time', 'Close']]
    df2['predict'] = prediction
    df2.to_csv('xxxvalidatepredicts.csv',  index=False)
    
if __name__ == "__main__":
    main()