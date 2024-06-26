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
import os

# Parametros necesarios: <interval> <live?> [limit] -> 5m True 289 (289 es el minimo para poder hacer prediccion para calcular los indicadores necesarios, el dropnulls eliminara 288 records que son 24hors en intervalos de 5m)
# Opcion1: Descargar datos live, (limitacion de 1000 registros) usando get_klines: python .\engineBTCLive.py 5m True 1000 
# Opcion2: Descargar datos historicos: python .\engineBTCLive.py 5m False

class GetDataBTC:
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        #api_key = os.getenv('BINANCE_API_KEY_TEST') # this key is for testnet
        #api_secret = os.getenv('BINANCE_API_SECRET_TEST') # this key is for tesnet


    def getBTCData(self, interval, live, limit):
        # interval = 1m,5m,1h,1d... (intervalo temporal, cada minuto)
        # loopback = 1 month, 10min etc... (poblacion)
        # If receive 1 , it will return a dataframe with the last record, otherwise a full dataset for model creation.
        # limit is the number of rows
        # if it is live it will ignore the loopback
        # if it is history it will ignore the limit
        #testnet
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        #api_key = os.getenv('BINANCE_API_KEY_TEST')
        #api_secret = os.getenv('BINANCE_API_SECRET_TEST')
        client = Client(api_key, api_secret)
        #client.API_URL = 'https://testnet.binance.vision/api' # testnet server
        client.API_URL = 'https://api.binance.com/api'
        symbol = 'BTCUSDT'
        #interval = Client.KLINE_INTERVAL_1MINUTE
        #loopback = '10 years'
        #df = pd.DataFrame(client.get_historical_klines(symbol, interval, loopback + ' ago UTC'))
        if live == True:
            # live data
            print("LOG: [ Live data run... !]")
            df = pd.DataFrame(client.get_klines(symbol=symbol, interval=interval, limit=limit))
            folder = "../data/live/"
        else:
            # history data
            print("LOG: [ History data run... !]")
            df = pd.DataFrame(client.get_historical_klines(symbol, interval, '10 years ago UTC'))
            folder = "../data/history/"
        # Asignar nombres a todas las columnas devueltas excluyendo 'Ignore'
        df.columns = [
            'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close Time', 'Quote Asset Volume', 'Number of Trades', 
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ]
        df.drop('Ignore', axis=1, inplace=True)
        df.drop('Close Time', axis=1, inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        df['Time'] = df['Time'].dt.tz_localize('UTC').dt.tz_convert('Europe/London').dt.strftime('%Y-%m-%d %H:%M:%S')
        #df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 
                    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']:
            df[col] = df[col].astype(float)
        #df=df.set_index('Time') # En caso de necesitar indice en el df.
        if live == True:
            df.to_csv(os.path.join(folder, 'live' + interval + '.csv'), index=False)
            print(f"LOG: [Live CSV Saved]: {folder}live{interval}.csv")
        else:
            df.to_csv(os.path.join(folder, 'fullhistory' + interval + '.csv'), index=False)
            print(f"LOG: [History CSV Saved]: {folder}fullhistory{interval}.csv")
        return df

def main():
    if len(sys.argv) < 3:
        print("ERROR: [ Parametros necesarios: <interval> <live?> [limit] ]: Por ejemplo: " + sys.argv[0] + " 5m " + "True " + "500 (limit solo se usa para limitar datos live) ")
        sys.exit(1) 
    btc = GetDataBTC()
    interval = sys.argv[1] # '5m'
    live = sys.argv[2].lower() == 'true'
    limit = 500 # binance te limita a 1000 en live.
    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
    df = btc.getBTCData(interval, live, limit)
    print(df[['Time', 'Close']].tail(3))
    #df[['Time', 'Close']].to_csv(('xxxxxx' + interval + '_up02pct.csv'), index=False)
     
if __name__ == "__main__":
    main()