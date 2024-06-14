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
import os

# No usar!
# Este esta obsoleto , el bueno es engineBTCLive.py que vale para la historia y para live.
# Mover proximamente a old.

class GetDataBTC:
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        #api_key = os.getenv('BINANCE_API_KEY_TEST')
        #api_secret = os.getenv('BINANCE_API_SECRET_TEST')


    def getBTCHistory(self, rows, interval, loopback, folder):
        # interval = 1m,5m,1h,1d... (intervalo temporal, cada minuto)
        # loopback = 1 month, 10min etc... (poblacion)
        #testnet
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        #api_key = os.getenv('BINANCE_API_KEY_TEST')
        #api_secret = os.getenv('BINANCE_API_SECRET_TEST')
        client = Client(api_key, api_secret)
        #client.API_URL = 'https://testnet.binance.vision/api'
        client.API_URL = 'https://api.binance.com/api'

        symbol = 'BTCUSDT'
        #interval = Client.KLINE_INTERVAL_1MINUTE
        #loopback = '10 years'
        df = pd.DataFrame(client.get_historical_klines(symbol, interval, loopback + ' ago UTC'))
        df.columns = [
            'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close Time', 'Quote Asset Volume', 'Number of Trades', 
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ]
        df.drop('Ignore', axis=1, inplace=True)
        df.drop('Close Time', axis=1, inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 
                    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']:
            df[col] = df[col].astype(float)
        df.to_csv(os.path.join(folder, 'fullhistory' + interval + '.csv'), index=False)
        print(f"LOG: [CSV Saved]: {folder}fullhistory{interval}.csv")
        if rows == 1:
            # I will need to do something in the code to get an enough df to do all the calculations but not all the history.
            return df.tail(1)
        else:
            return df
    
    



